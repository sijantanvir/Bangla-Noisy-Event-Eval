import argparse
import json
import os
import sys

import torch
from datasets import load_dataset
from seqeval.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorForTokenClassification

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import EncoderTokenClassifier


def tokenize_and_align(examples, tokenizer, label2id, max_length=128):
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        max_length=max_length,
        is_split_into_words=True,
    )
    aligned_labels = []
    for i, labels in enumerate(examples["labels"]):
        word_ids = tokenized.word_ids(batch_index=i)
        prev_word_id = None
        label_ids = []
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != prev_word_id:
                label_ids.append(label2id.get(labels[word_id], label2id["O"]))
            else:
                label_ids.append(-100)
            prev_word_id = word_id
        aligned_labels.append(label_ids)
    tokenized["labels"] = aligned_labels
    return tokenized


def evaluate(test_file, model_dir, label_info, device, batch_size=32, max_length=128):
    label2id = label_info["label2id"]
    id2label = {int(k): v for k, v in label_info["id2label"].items()}
    num_labels = len(id2label)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = EncoderTokenClassifier(model_dir, num_labels)

    state_dict = torch.load(
        os.path.join(model_dir, "pytorch_model.bin"),
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    test_ds = load_dataset("json", data_files=test_file, split="train")
    test_tok = test_ds.map(
        lambda x: tokenize_and_align(x, tokenizer, label2id, max_length),
        batched=True,
        remove_columns=test_ds.column_names,
        load_from_cache_file=False,
    )

    collator = DataCollatorForTokenClassification(tokenizer)
    loader = DataLoader(test_tok, batch_size=batch_size, collate_fn=collator)

    true_preds, true_labels = [], []

    for batch in tqdm(loader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].tolist()

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs["logits"].argmax(dim=-1).tolist()

        for i in range(len(preds)):
            cp, cl = [], []
            for p, l in zip(preds[i], labels[i]):
                if l != -100:
                    if p == -100:
                        p = 0
                    cp.append(id2label[p])
                    cl.append(id2label[l])
            true_preds.append(cp)
            true_labels.append(cl)

    micro_p = precision_score(true_labels, true_preds)
    micro_r = recall_score(true_labels, true_preds)
    micro_f1 = f1_score(true_labels, true_preds)
    macro_f1 = f1_score(true_labels, true_preds, average="macro")

    generic_true = [
        ["B-EVENT" if l.startswith("B-") else "I-EVENT" if l.startswith("I-") else "O" for l in seq]
        for seq in true_labels
    ]
    generic_pred = [
        ["B-EVENT" if p.startswith("B-") else "I-EVENT" if p.startswith("I-") else "O" for p in seq]
        for seq in true_preds
    ]
    det_f1 = f1_score(generic_true, generic_pred)

    return {
        "strict": {
            "micro_p": round(micro_p * 100, 2),
            "micro_r": round(micro_r * 100, 2),
            "micro_f1": round(micro_f1 * 100, 2),
            "macro_f1": round(macro_f1 * 100, 2),
        },
        "trigger_id_f1": round(det_f1 * 100, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    args = parser.parse_args()

    label_info = json.load(open(os.path.join(args.model_dir, "label_info.json")))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    results = evaluate(
        args.test_file, args.model_dir, label_info, device,
        batch_size=args.batch_size, max_length=args.max_length,
    )

    print(f"\nStrict Macro-F1: {results['strict']['macro_f1']}%")
    print(f"Trigger ID F1: {results['trigger_id_f1']}%")

    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
