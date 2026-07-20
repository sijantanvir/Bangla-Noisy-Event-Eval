import argparse
import json
import os
import sys

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import build_label_space, tokenize_encoder
from src.metrics import compute_metrics
from src.models import EncoderTokenClassifier


class EncoderTrainer(Trainer):
    def save_model(self, output_dir=None, _internal_call=False):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.model.state_dict(), f"{output_dir}/pytorch_model.bin")
        self.model.encoder.config.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        label_info = {
            "label_list": self.label_list,
            "label2id": self.label2id,
            "id2label": self.id2label,
        }
        with open(f"{output_dir}/label_info.json", "w") as f:
            json.dump(label_info, f)

        custom_config = {
            "model_name": self.model_name,
            "num_labels": len(self.label_list),
        }
        with open(f"{output_dir}/custom_config.json", "w") as f:
            json.dump(custom_config, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="csebuetnlp/banglabert")
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--val_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dataset = load_dataset("json", data_files={"train": args.train_file, "validation": args.val_file})
    train_ds = dataset["train"]
    val_ds = dataset["validation"]

    label_list, label2id, id2label = build_label_space(train_ds)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_tok = train_ds.map(
        lambda x: tokenize_encoder(x, tokenizer, label2id, args.max_length),
        batched=True,
        remove_columns=train_ds.column_names,
    )
    val_tok = val_ds.map(
        lambda x: tokenize_encoder(x, tokenizer, label2id, args.max_length),
        batched=True,
        remove_columns=val_ds.column_names,
    )

    model = EncoderTokenClassifier(
        model_name=args.model_name,
        num_labels=len(label_list),
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        fp16=torch.cuda.is_available(),
        report_to="none",
        optim="adamw_torch",
        warmup_ratio=0.1,
    )

    trainer = EncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=lambda x: compute_metrics(x, id2label),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )
    trainer.model_name = args.model_name
    trainer.label_list = label_list
    trainer.label2id = label2id
    trainer.id2label = id2label
    trainer.tokenizer = tokenizer

    print(f"Training {args.model_name}")
    trainer.train()

    final_dir = f"{args.output_dir}/final"
    trainer.save_model(final_dir)
    print(f"Saved to {final_dir}")


if __name__ == "__main__":
    main()
