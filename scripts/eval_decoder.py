import argparse
import json
import os
import re
import sys

import torch
from tqdm import tqdm
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

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




def extract_mention(text, event_type):
    event_name = event_type.split("_")[-1]
    pattern = rf'{event_name}\((mention="([^"]*)"|mention=\'([^\']*)\')\)'
    match = re.search(pattern, text)
    if match:
        return match.group(1) or match.group(2)
    return None


def build_prompt(sentence, event_type, use_guide=True):
    event_name = event_type.split("_")[-1].replace("-", "")
    if use_guide:
        schema_block = SCHEMAS.get(event_name, f"@dataclass\nclass {event_name}({event_type.split('_')[0]}):\n    mention: str\n")
    else:
        schema_block = f"class {event_name}:\n    mention: str\n"

    prompt = (
        f"# Instruction\nExtract events according to the schema.\n\n"
        f"# Schema\n{schema_block}\n"
        f"# Sentence\ntext = \"{sentence}\"\n\nresult ="
    )
    return prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="unsloth/Llama-3.2-1B-Instruct")
    parser.add_argument("--lora_path", required=True)
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--use_guide", action="store_true")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    print(f"Loading {args.model_name} with LoRA from {args.lora_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_length,
        dtype=None,
        load_in_4bit=False,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    model.load_adapter(args.lora_path)
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

    with open(args.test_file, "r", encoding="utf-8") as f:
        test_data = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(test_data)} sentences. Running one-vs-all inference...")

    tp_by_type = {et: 0 for et in EVENT_TYPES}
    fp_by_type = {et: 0 for et in EVENT_TYPES}
    fn_by_type = {et: 0 for et in EVENT_TYPES}

    for item in tqdm(test_data):
        sentence = " ".join(item["tokens"])
        labels = item["labels"]

        gold_events = set()
        for lbl in labels:
            if lbl != "O":
                et = lbl.replace("B-", "").replace("I-", "")
                gold_events.add(et)

        for event_type in EVENT_TYPES:
            prompt = build_prompt(sentence, event_type, args.use_guide)

            messages = [
                {"role": "user", "content": prompt},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False)

            inputs = tokenizer([text], return_tensors="pt", truncation=True,
                               max_length=args.max_length).to("cuda")

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    temperature=0.0,
                    do_sample=False,
                )

            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

            predicted = extract_mention(response, event_type)
            in_gold = event_type in gold_events

            if predicted is not None and in_gold:
                tp_by_type[event_type] += 1
            elif predicted is not None and not in_gold:
                fp_by_type[event_type] += 1
            elif predicted is None and in_gold:
                fn_by_type[event_type] += 1

    class_f1s = []
    for et in EVENT_TYPES:
        tp = tp_by_type[et]
        fp = fp_by_type[et]
        fn = fn_by_type[et]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        class_f1s.append(f1)

    macro_f1 = sum(class_f1s) / len(class_f1s) * 100
    print(f"\nMacro-F1: {macro_f1:.2f}%")

    results = {"macro_f1": round(macro_f1, 2)}
    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {args.output_file}")


if __name__ == "__main__":
    main()
