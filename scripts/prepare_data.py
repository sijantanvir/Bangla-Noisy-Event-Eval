import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_jsonl, rarest_first_split, print_distribution, save_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="HF JSONL (tokens+labels)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mode", choices=["news", "asr"], required=True)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    args = parser.parse_args()

    data = load_jsonl(args.input)
    prefix = "clean" if args.mode == "news" else "asr"

    print(f"Splitting {prefix} data ({len(data)} sentences)...")
    train, val, test = rarest_first_split(data, args.train_ratio, args.val_ratio, args.test_ratio)

    split_dir = os.path.join(args.output_dir, f"{prefix}")
    os.makedirs(split_dir, exist_ok=True)

    save_jsonl(train, os.path.join(split_dir, "train.jsonl"))
    save_jsonl(val, os.path.join(split_dir, "val.jsonl"))
    save_jsonl(test, os.path.join(split_dir, "test.jsonl"))

    print_distribution(train, val, test)
    print(f"\nSaved to {split_dir}/")
    print(f"  Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")


if __name__ == "__main__":
    main()
