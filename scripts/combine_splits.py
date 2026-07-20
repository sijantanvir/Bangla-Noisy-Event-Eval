import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_dir", required=True)
    parser.add_argument("--asr_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", choices=["train", "val"], default="train")
    args = parser.parse_args()

    clean_file = os.path.join(args.clean_dir, f"{args.split}.jsonl")
    asr_file = os.path.join(args.asr_dir, f"{args.split}.jsonl")
    os.makedirs(args.output_dir, exist_ok=True)

    combined = []
    for fpath in [clean_file, asr_file]:
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    combined.append(json.loads(line))

    out_file = os.path.join(args.output_dir, f"{args.split}.jsonl")
    with open(out_file, "w", encoding="utf-8") as f:
        for entry in combined:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Combined {args.split}: {len(combined)} sentences -> {out_file}")


if __name__ == "__main__":
    main()
