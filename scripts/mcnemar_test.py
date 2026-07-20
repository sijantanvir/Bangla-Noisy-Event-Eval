import json
import os
import re
import sys

import pandas as pd
from difflib import SequenceMatcher
from statsmodels.stats.contingency_tables import mcnemar
from seqeval.metrics.sequence_labeling import get_entities

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT_DIR = os.environ.get("MODELS_DIR", "models")

TARGET_ENCODERS = [
    'banglabert_dsoftmax', 'banglabert_large_dsoftmax',
    'xlm-roberta-base_dsoftmax', 'xlm-roberta-large_dsoftmax'
]

TARGET_LLMS = [
    'gemma3_4B_16bit', 'llama3_1_8b_16bit_r16', 'llama3_2_1b_16bit', 'gemma3_1b_16bit'
]
WCR_LEVELS = [10, 20, 30, 40]

LLM_DISPLAY_NAMES = {
    'gemma3_4B_16bit': 'Gemma3 4B IT',
    'llama3_1_8b_16bit_r16': 'Llama3.1 8B IT',
    'llama3_2_1b_16bit': 'Llama3.2 1B IT',
    'gemma3_1b_16bit': 'Gemma3 1B IT',
}


def find_best_fuzzy_match(mention_text, original_tokens, used_indices, threshold=0.7):
    mention_toks = mention_text.strip().split()
    n = len(mention_toks)
    if n == 0:
        return None

    best_match_idx, best_score = None, 0.0
    for i in range(len(original_tokens) - n + 1):
        if any(idx in used_indices for idx in range(i, i + n)):
            continue
        window_text = " ".join(original_tokens[i:i + n])
        score = SequenceMatcher(None, mention_text, window_text).ratio()
        if score > best_score:
            best_score = score
            best_match_idx = i

    if best_score >= threshold:
        return best_match_idx, best_match_idx + n - 1
    return None


def normalize_to_camelcase(raw_type):
    if "_" in raw_type:
        event_tail = raw_type.split("_", 1)[1]
    else:
        event_tail = raw_type
    return "".join([p.capitalize() for p in event_tail.split("-") if p])


def find_files():
    print("Scanning for model prediction files...")

    encoder_files = {model: {} for model in TARGET_ENCODERS}
    llm_files = {model: {"wGuide": {}, "w/oGuide": {}} for model in TARGET_LLMS}

    for root, dirs, files in os.walk(ROOT_DIR):
        norm_root = root.replace("\\", "/")

        if any(bad_dir in norm_root.lower() for bad_dir in ["ignore", "legacy", "old_wcr", "combined"]):
            continue

        for file in files:
            if "predictions_synthetic_test_wcr_" in file and file.endswith(".json"):
                for enc in TARGET_ENCODERS:
                    if enc in norm_root:
                        wcr_match = re.search(r"wcr_(\d+)", file)
                        if wcr_match:
                            wcr = int(wcr_match.group(1))
                            if wcr in WCR_LEVELS:
                                encoder_files[enc][wcr] = os.path.join(root, file)
                        break

            elif file.endswith("predictions_final.jsonl"):
                for llm in TARGET_LLMS:
                    if llm in norm_root:
                        wcr_match = re.search(r"wcr_(\d+)", file)
                        if wcr_match:
                            wcr = int(wcr_match.group(1))
                            g_status = "w/oGuide" if "guideline_free" in norm_root.lower() else "wGuide"
                            if wcr in WCR_LEVELS:
                                llm_files[llm][g_status][wcr] = os.path.join(root, file)
                        break

    return encoder_files, llm_files


def run_all_comparisons():
    encoder_files, llm_files = find_files()

    print("\nFile Discovery Summary:")
    for enc in TARGET_ENCODERS:
        found = list(encoder_files[enc].keys())
        if found:
            print(f"  {enc}: WCR levels found = {sorted(found)}")

    for llm in TARGET_LLMS:
        for g_status in ["wGuide", "w/oGuide"]:
            found = list(llm_files[llm][g_status].keys())
            if found:
                print(f"  {llm} ({g_status}): WCR levels found = {sorted(found)}")

    print("\n" + "=" * 50)

    results_data = []

    for wcr in WCR_LEVELS:
        print(f"Processing WCR {wcr}%...")

        llm_parsed_data = {}
        for llm in TARGET_LLMS:
            for g_status in ["wGuide", "w/oGuide"]:
                if wcr not in llm_files[llm][g_status]:
                    continue

                sentence_map = {}
                with open(llm_files[llm][g_status][wcr], 'r', encoding='utf-8') as f:
                    for line in f:
                        if not line.strip():
                            continue
                        data = json.loads(line)
                        tokens = data.get("tokens", [])
                        token_key = " ".join(tokens)
                        schema = data.get("target_schema", "")

                        if token_key not in sentence_map:
                            sentence_map[token_key] = {
                                "is_trigger_corrupt": data.get("sentence_trigger_corrupted", False),
                                "is_context_corrupt": any(data.get("token_corrupted_flags", []))
                                    and not data.get("sentence_trigger_corrupted", False),
                                "llm_strict_ents": set()
                            }

                        pred_text = data.get("predicted_output", "")
                        pattern = rf'{schema}\(mention=["\'](.*?)["\']\)'
                        predicted_mentions = re.findall(pattern, pred_text)

                        used_indices = set()
                        for mention in predicted_mentions:
                            match_indices = find_best_fuzzy_match(mention, tokens, used_indices)
                            if match_indices:
                                start_idx, end_idx = match_indices
                                sentence_map[token_key]["llm_strict_ents"].add((schema, start_idx, end_idx))
                                used_indices.update(range(start_idx, end_idx + 1))

                llm_parsed_data[(llm, g_status)] = sentence_map

        for enc in TARGET_ENCODERS:
            if wcr not in encoder_files[enc]:
                continue

            with open(encoder_files[enc][wcr], 'r', encoding='utf-8') as f:
                bert_data = json.load(f)

            for (llm, g_status) in llm_parsed_data.keys():
                trigger_pairs, context_pairs = [], []
                llm_map = llm_parsed_data[(llm, g_status)]

                for row in bert_data:
                    if not isinstance(row, dict):
                        continue
                    token_key = " ".join(row.get("tokens", []))
                    if token_key not in llm_map:
                        continue

                    llm_record = llm_map[token_key]
                    is_trig = llm_record["is_trigger_corrupt"]
                    is_ctx = llm_record["is_context_corrupt"]

                    if not is_trig and not is_ctx:
                        continue

                    t_labels = row.get("true_labels", [])
                    gold_ents = {(normalize_to_camelcase(t), s, e) for t, s, e in get_entities(t_labels)}

                    if len(gold_ents) == 0:
                        continue

                    p_labels = row.get("predicted_labels", [])
                    bert_ents = {(normalize_to_camelcase(t), s, e) for t, s, e in get_entities(p_labels)}
                    llm_ents = llm_record["llm_strict_ents"]

                    bert_correct = 1 if len(gold_ents & bert_ents) > 0 else 0
                    llm_correct = 1 if len(gold_ents & llm_ents) > 0 else 0

                    if is_trig:
                        trigger_pairs.append((bert_correct, llm_correct))
                    elif is_ctx:
                        context_pairs.append((bert_correct, llm_correct))

                def calc_mcnemar(pairs):
                    if not pairs:
                        return 0, 0, 1.0, 0, 0
                    n10 = sum(1 for b, l in pairs if b == 1 and l == 0)
                    n01 = sum(1 for b, l in pairs if b == 0 and l == 1)
                    discordant = n10 + n01

                    if discordant == 0:
                        return len(pairs), discordant, 1.0, n10, n01

                    res = mcnemar(
                        [[sum(1 for b, l in pairs if b == 1 and l == 1), n10],
                         [n01, sum(1 for b, l in pairs if b == 0 and l == 0)]],
                        exact=True
                    )
                    return len(pairs), discordant, res.pvalue, n10, n01

                t_n, t_disc, t_pval, t_enc_won, t_llm_won = calc_mcnemar(trigger_pairs)
                c_n, c_disc, c_pval, c_enc_won, c_llm_won = calc_mcnemar(context_pairs)

                clean_llm_name = LLM_DISPLAY_NAMES.get(llm, llm)
                display_enc = enc.replace("_dsoftmax", "")

                results_data.append({
                    "WCR": f"WCR {wcr}",
                    "Encoder": display_enc,
                    "LLM": f"{clean_llm_name} ({g_status})",
                    "Trig N": t_n,
                    "T_BERT_wins": t_enc_won,
                    "T_LLM_wins": t_llm_won,
                    "Trig Sig (p-value)": t_pval,
                    "Ctx N": c_n,
                    "C_BERT_wins": c_enc_won,
                    "C_LLM_wins": c_llm_won,
                    "Ctx Sig (p-value)": c_pval,
                })

    df = pd.DataFrame(results_data)
    if df.empty:
        print("No valid comparisons were made. Check file paths.")
        return

    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plots", "data", "All_Models_Significance_Strict.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    pd.options.display.float_format = '{:.4f}'.format
    display_cols = ["WCR", "Encoder", "LLM", "T_BERT_wins", "T_LLM_wins",
                    "Trig Sig (p-value)", "C_BERT_wins", "C_LLM_wins", "Ctx Sig (p-value)"]
    print("\n" + "=" * 120)
    print("STATISTICAL SIGNIFICANCE SUMMARY (STRICT EXACT MATCH)")
    print("=" * 120)
    print(df[display_cols].to_string(index=False))


if __name__ == "__main__":
    run_all_comparisons()
