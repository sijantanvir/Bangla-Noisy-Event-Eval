import argparse
import json
import os
import random
import sys

import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import FastLanguageModel, is_bf16_supported
from unsloth.chat_templates import get_chat_template, train_on_responses_only

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_event_schemas

SCHEMAS = load_event_schemas(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Dataset", "schema.md")
)

EVENT_TYPES = [
    "Contact_Meet", "Contact_Phone-Write", "Conflict_Attack", "Conflict_Demonstrate",
    "Crime_Commit-Blue-Collar-Crime", "Crime_Commit-White-Collar-Crime",
    "Disaster_Occur-Man-Made-Disaster", "Disaster_Occur-Natural-Disaster",
    "Festival_Celebrate-Cultural-Festival", "Festival_Observe-Religious-Festival",
    "Governance_Approve", "Governance_Ban", "Governance_Decide", "Governance_Gazette",
    "Governance_Investigate", "Governance_Reform", "Governance_Support",
    "Health_Outbreak", "Health_Serve-Patients",
    "Justice_Arrest-Jail", "Justice_Charge-Indict", "Justice_Deliver-Verdict",
    "Justice_Sue", "Justice_Trial-Hearing",
    "Life_Die", "Life_Injure", "Movement_Transport",
    "Personnel_Elect", "Personnel_End-Position", "Personnel_Start-Position",
    "Socio-Economy_Disrupt", "Socio-Economy_Graduate", "Socio-Economy_Grow",
    "Socio-Economy_Import-Export", "Socio-Economy_Invest", "Socio-Economy_Recruit",
    "Socio-Economy_Trade", "Technology_Launch-Service",
    "Transaction_Transfer-Money", "Transaction_Transfer-Ownership",
]


def build_prompt(event_type, use_guide=True):
    event_name = event_type.split("_")[-1].replace("-", "")
    if use_guide:
        schema_block = SCHEMAS.get(event_name)
        if schema_block:
            return f"# Instruction\nExtract events according to the schema.\n\n# Schema\n{schema_block}\n\n"
    return (
        f"# Instruction\nExtract events according to the schema.\n\n"
        f"# Schema\nclass {event_name}:\n    mention: str\n\n"
    )


def collate_messages(examples, tokenizer):
    texts = []
    for i in range(len(examples["tokens"])):
        ex = {k: examples[k][i] for k in examples}

        tokens = ex["tokens"]
        labels = ex["labels"]
        sentence = " ".join(tokens)

        events_in_sentence = set()
        for lbl in labels:
            if lbl != "O":
                events_in_sentence.add(lbl.replace("B-", "").replace("I-", ""))

        all_types = list(EVENT_TYPES)
        pos = list(events_in_sentence)
        neg = [e for e in all_types if e not in events_in_sentence]
        random.shuffle(neg)
        queries = pos + neg[:5]
        random.shuffle(queries)

        messages = []
        for et in queries:
            prompt = build_prompt(et, True)
            prompt += f'# Sentence\ntext = "{sentence}"\n\nresult ='

            if et in events_in_sentence:
                event_name = et.split("_")[-1]
                for lbl, tok in zip(labels, tokens):
                    if lbl.startswith("B-") and lbl.replace("B-", "").replace("I-", "") == et:
                        result = f'[\n    {event_name}(mention="{tok}")\n]'
                        break
                else:
                    result = "[]"
            else:
                result = "[]"

            messages.append({"role": "user", "content": prompt})
            messages.append({"role": "assistant", "content": result})

        texts.append(tokenizer.apply_chat_template(messages, tokenize=False))
    return {"text": texts}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="unsloth/Llama-3.2-1B-Instruct")
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--val_file", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--format", choices=["raw", "messages", "auto"], default="auto")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--per_device_batch_size", type=int, default=8)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(42)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_length,
        dtype=None,
        load_in_4bit=False,
    )

    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    data_files = {"train": args.train_file}
    if args.val_file:
        data_files["validation"] = args.val_file
    dataset = load_dataset("json", data_files=data_files)

    has_messages = "messages" in dataset["train"].features
    if args.format == "messages" or (args.format == "auto" and has_messages):
        print("Using pre-formatted messages format")

        def apply_template(examples):
            return {"text": [
                tokenizer.apply_chat_template(msgs, tokenize=False)
                for msgs in examples["messages"]
            ]}

        train_ds = dataset["train"].map(apply_template, batched=True, remove_columns=dataset["train"].column_names)
        if args.val_file:
            val_ds = dataset["validation"].map(apply_template, batched=True, remove_columns=dataset["validation"].column_names)
        else:
            val_ds = None
    else:
        print("Formatting from raw tokens/labels")
        train_ds = dataset["train"].map(
            lambda x: collate_messages(x, tokenizer),
            batched=True,
            input_columns=dataset["train"].column_names,
            remove_columns=dataset["train"].column_names,
        )
        if args.val_file:
            val_ds = dataset["validation"].map(
                lambda x: collate_messages(x, tokenizer),
                batched=True,
                input_columns=dataset["validation"].column_names,
                remove_columns=dataset["validation"].column_names,
            )
        else:
            val_ds = None

    total_steps = len(train_ds) * args.epochs
    warmup_steps = int(total_steps * 0.05)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.batch_size // args.per_device_batch_size,
        num_train_epochs=args.epochs,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        logging_steps=50,
        eval_strategy="steps" if val_ds else "no",
        eval_steps=0.2 if val_ds else None,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True if val_ds else False,
        metric_for_best_model="eval_loss" if val_ds else None,
        fp16=not is_bf16_supported(),
        bf16=is_bf16_supported(),
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=training_args,
        max_seq_length=args.max_length,
        dataset_text_field="text",
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>",
        response_part="<|start_header_id|>assistant<|end_header_id|>",
    )

    print(f"Train: {len(train_ds)} | Val: {len(val_ds) if val_ds else 0}")
    trainer.train()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
