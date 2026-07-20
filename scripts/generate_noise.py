import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.noise import BengaliErrorGenerator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--wcr_levels", nargs="+", type=int, default=[10, 20, 30, 40])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)

    clean_data = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                clean_data.append(json.loads(line))

    progressive_indices = []
    for item in clean_data:
        indices = list(range(len(item["tokens"])))
        random.shuffle(indices)
        progressive_indices.append(indices)

    error_generator = BengaliErrorGenerator()

    for wcr in args.wcr_levels:
        output_file = os.path.join(args.output_dir, f"test_wcr{wcr}.jsonl")

        synthetic_dataset = []
        total_tokens = 0
        targeted_count = 0
        changed_count = 0

        for item, shuffled_idx in zip(clean_data, progressive_indices):
            original_tokens = item["tokens"]
            labels = item["labels"]
            num_tokens = len(original_tokens)

            if wcr == 0:
                k_target = 0
            else:
                k_target = max(1, round(num_tokens * wcr / 100))
                k_target = min(k_target, num_tokens)

            targeted_indices = set(shuffled_idx[:k_target])

            noisy_tokens = []
            corrupted_flags = []
            trigger_was_corrupted = False

            for idx, (tok, lbl) in enumerate(zip(original_tokens, labels)):
                total_tokens += 1
                if idx in targeted_indices:
                    targeted_count += 1
                    mutated = error_generator.corrupt_word(tok)
                    noisy_tokens.append(mutated)
                    if mutated != tok:
                        changed_count += 1
                        corrupted_flags.append(True)
                        if lbl != "O":
                            trigger_was_corrupted = True
                    else:
                        corrupted_flags.append(False)
                else:
                    noisy_tokens.append(tok)
                    corrupted_flags.append(False)

            synthetic_dataset.append({
                "tokens": noisy_tokens,
                "original_tokens": original_tokens,
                "labels": labels,
                "token_corrupted_flags": corrupted_flags,
                "sentence_trigger_corrupted": trigger_was_corrupted,
            })

        with open(output_file, "w", encoding="utf-8") as f:
            for entry in synthetic_dataset:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        targeted_pct = (targeted_count / total_tokens) * 100
        actual_pct = (changed_count / total_tokens) * 100
        print(f"WCR {wcr}% | Targeted: {targeted_pct:.2f}% | Corrupted: {actual_pct:.2f}% | Saved: {output_file}")

    print(f"\nDone. Generated {len(args.wcr_levels)} noisy test sets in {args.output_dir}")


if __name__ == "__main__":
    main()
