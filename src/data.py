import json
import os
import random
import re
from collections import Counter


def load_jsonl(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            return json.load(f)
        else:
            return [json.loads(line) for line in f if line.strip()]


def save_jsonl(data, filename):
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        for ex in data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def build_label_space(dataset):
    label_set = {"O"}
    for ex in dataset:
        label_set.update(ex["labels"])
    new_i = {l.replace("B-", "I-") for l in label_set if l.startswith("B-")}
    label_set.update(new_i)
    label_list = ["O"] + sorted([l for l in label_set if l != "O"])
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}
    return label_list, label2id, id2label


def tokenize_encoder(examples, tokenizer, label2id, max_length=128):
    from normalizer import normalize
    normalized_tokens = [[normalize(tok) for tok in seq] for seq in examples["tokens"]]
    tokenized = tokenizer(
        normalized_tokens,
        is_split_into_words=True,
        padding="max_length",
        truncation=True,
        max_length=max_length,
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
                label = labels[word_id]
                if label.startswith("B-"):
                    label = label.replace("B-", "I-")
                label_ids.append(label2id.get(label, label2id["O"]))
            prev_word_id = word_id
        aligned_labels.append(label_ids)

    tokenized["labels"] = aligned_labels
    return tokenized


def rarest_first_split(data, train_ratio, val_ratio, test_ratio, seed=42):
    random.seed(seed)

    global_event_counts = Counter()
    for idx, ex in enumerate(data):
        ex["_idx"] = idx
        events = set(
            lbl.replace("B-", "").replace("I-", "")
            for lbl in ex["labels"]
            if lbl != "O"
        )
        ex["_events"] = events
        for e in events:
            global_event_counts[e] += 1

    sorted_events = [e for e, _ in global_event_counts.most_common()[::-1]]

    train_vault, val_vault, test_vault = [], [], []
    used_indices = set()

    for event_type in sorted_events:
        candidates = [
            ex for ex in data
            if event_type in ex["_events"] and ex["_idx"] not in used_indices
        ]
        if not candidates:
            continue
        random.shuffle(candidates)
        n = len(candidates)

        if n == 1:
            print(f"  Warning: '{event_type}' has 1 sentence -> train only")
            train_vault.append(candidates[0])
        elif n == 2:
            train_vault.append(candidates[0])
            test_vault.append(candidates[1])
        elif n == 3:
            train_vault.append(candidates[0])
            val_vault.append(candidates[1])
            test_vault.append(candidates[2])
        else:
            train_end = int(n * train_ratio)
            val_end = int(n * (train_ratio + val_ratio))
            train_end = max(1, min(train_end, n - 2))
            val_end = max(train_end + 1, min(val_end, n - 1))
            train_vault.extend(candidates[:train_end])
            val_vault.extend(candidates[train_end:val_end])
            test_vault.extend(candidates[val_end:])

        for ex in candidates:
            used_indices.add(ex["_idx"])

    negative = [ex for ex in data if ex["_idx"] not in used_indices]
    random.shuffle(negative)
    n_neg = len(negative)
    neg_train_end = int(n_neg * train_ratio)
    neg_val_end = int(n_neg * (train_ratio + val_ratio))
    train_vault.extend(negative[:neg_train_end])
    val_vault.extend(negative[neg_train_end:neg_val_end])
    test_vault.extend(negative[neg_val_end:])

    def cleanup(vault):
        for ex in vault:
            ex.pop("_idx", None)
            ex.pop("_events", None)
        random.shuffle(vault)
        return vault

    return cleanup(train_vault), cleanup(val_vault), cleanup(test_vault)


def print_distribution(train, val, test):
    def get_dist(vault):
        counts = Counter()
        for ex in vault:
            for lbl in ex["labels"]:
                if lbl != "O" and lbl.startswith("B-"):
                    counts[lbl.replace("B-", "")] += 1
        return counts

    td = get_dist(train)
    vd = get_dist(val)
    tsd = get_dist(test)

    all_events = sorted(set(list(td.keys()) + list(vd.keys()) + list(tsd.keys())))
    print(f"\n{'Event Type':<40} {'Train':>6} {'Val':>6} {'Test':>6}")
    print("-" * 60)
    for e in all_events:
        print(f"{e:<40} {td.get(e, 0):>6} {vd.get(e, 0):>6} {tsd.get(e, 0):>6}")
    print("-" * 60)
    print(f"{'TOTAL':<40} {len(train):>6} {len(val):>6} {len(test):>6}")


def load_event_schemas(schema_path):
    with open(schema_path, "r", encoding="utf-8") as f:
        text = f.read()
    pattern = re.compile(
        r"(@dataclass\s+class\s+([A-Za-z]+)\(.*?\):\s+"
        r'"""(?:.*?)"""\s+mention:\s+str)',
        re.DOTALL,
    )
    return {m.group(2): m.group(1).strip() for m in pattern.finditer(text)}
