import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from collections import OrderedDict

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 17,
    'axes.labelsize': 13,
    'ytick.labelsize': 13,
})

ROOT = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(ROOT, 'data', 'Clean_Test_Results.csv'), 'r') as f:
    clean_rows = list(csv.DictReader(f))
with open(os.path.join(ROOT, 'data', 'ASR_Test_Results.csv'), 'r') as f:
    asr_rows = list(csv.DictReader(f))
with open(os.path.join(ROOT, 'data', 'WCR_Test_Results.csv'), 'r') as f:
    wcr_rows = list(csv.DictReader(f))


def _alias(model):
    return 'llama3_1_8b_16bit_r16' if model == 'lamma3_1_8b_16bit_r16' else model


clean_rows = [{**r, 'Model': _alias(r['Model'])} for r in clean_rows]
asr_rows = [{**r, 'Model': _alias(r['Model'])} for r in asr_rows]
wcr_rows = [{**r, 'Model': _alias(r['Model'])} for r in wcr_rows]

FAMILY_CMAP = {
    'Gemma': plt.cm.Blues,
    'LLaMA': plt.cm.Oranges,
    'BanglaBERT': plt.cm.Reds,
    'XLM-R': plt.cm.Greens,
}

MODELS = [
    ('Gemma3 1B IT', 'gemma3_1b_16bit', 'Gemma', [
        ('Clean', 'Guided', 'Clean+w/ Guide'),
        ('Combined', 'Guided', 'Comb.+w/ Guide'),
        ('Clean', 'Guideline-Free', 'Clean+w/o Guide'),
    ]),
    ('Gemma3 4B IT', 'gemma3_4B_16bit', 'Gemma', [
        ('Clean', 'Guided', 'Clean+w/ Guide'),
        ('Combined', 'Guided', 'Comb.+w/ Guide'),
        ('Clean', 'Guideline-Free', 'Clean+w/o Guide'),
    ]),
    ('Llama3.2 1B IT', 'llama3_2_1b_16bit', 'LLaMA', [
        ('Clean', 'Guided', 'Clean+w/ Guide'),
        ('Combined', 'Guided', 'Comb.+w/ Guide'),
        ('Clean', 'Guideline-Free', 'Clean+w/o Guide'),
    ]),
    ('Llama3.1 8B IT', 'llama3_1_8b_16bit_r16', 'LLaMA', [
        ('Clean', 'Guided', 'Clean+w/ Guide'),
        ('Combined', 'Guided', 'Comb.+w/ Guide'),
        ('Clean', 'Guideline-Free', 'Clean+w/o Guide'),
    ]),
    ('BanglaBERT', 'banglabert_dsoftmax', 'BanglaBERT', [
        ('Clean', 'N/A', 'Clean'),
        ('Combined', 'N/A', 'Comb.'),
    ]),
    ('BanglaBERT-L', 'banglabert_large_dsoftmax', 'BanglaBERT', [
        ('Clean', 'N/A', 'Clean'),
        ('Combined', 'N/A', 'Comb.'),
    ]),
    ('XLM-R', 'xlm-roberta-base_dsoftmax', 'XLM-R', [
        ('Clean', 'N/A', 'Clean'),
        ('Combined', 'N/A', 'Comb.'),
    ]),
    ('XLM-R-Large', 'xlm-roberta-large_dsoftmax', 'XLM-R', [
        ('Clean', 'N/A', 'Clean'),
        ('Combined', 'N/A', 'Comb.'),
    ]),
]


def get_clean_macro(model, train, guide):
    matches = [r for r in clean_rows
               if r['Model'] == model and r['Training_Data'] == train and r['Guideline'] == guide]
    return float(matches[0]['Strict_Macro_F1']) if matches else None


def get_asr_macro(model, train, guide):
    matches = [r for r in asr_rows
               if r['Model'] == model and r['Training_Data'] == train and r['Guideline'] == guide]
    return float(matches[0]['Strict_Macro_F1']) if matches else None


def get_avg_wcr_macro(model, train, guide):
    vals = []
    for lv in [10, 20, 30, 40]:
        matches = [r for r in wcr_rows
                    if r['Model'] == model and r['Training_Data'] == train
                    and r['Guideline'] == guide and r['WCR_Level'] == str(lv)]
        if matches:
            vals.append(float(matches[0]['Strict_Macro_F1']))
    return np.mean(vals) if vals else None


def chart():
    bar_width = 0.72
    inter_model_gap = 0.75
    inter_family_gap = 0.6

    xs = []
    retained_list = []
    drop_list = []
    clean_list = []
    colors_list = []
    config_labels = []
    model_spans = []
    family_spans = OrderedDict()

    pos = 0
    prev_family = None
    prev_model_key = None
    config_counter = {}

    for model_name, model_key, family, configs in MODELS:
        if prev_family is not None and family != prev_family:
            pos += inter_family_gap
        elif prev_model_key is not None:
            pos += inter_model_gap

        cmap = FAMILY_CMAP[family]
        n_total = sum(len(c) for _, _, f, c in MODELS if f == family)
        config_counter.setdefault(family, 0)
        ci = config_counter[family]

        group_start = pos
        group_xs = []

        for train, guide, cfg_label in configs:
            clean_f1 = get_clean_macro(model_key, train, guide)
            asr_f1 = get_asr_macro(model_key, train, guide)
            avg_wcr = get_avg_wcr_macro(model_key, train, guide)

            if clean_f1 is None or asr_f1 is None or avg_wcr is None:
                pos += 1
                continue

            retained = (asr_f1 + avg_wcr) / 2.0
            noise_drop = clean_f1 - retained

            shade = 0.38 + 0.30 * (ci / max(n_total - 1, 1))
            color = cmap(shade)

            xs.append(pos)
            retained_list.append(retained)
            drop_list.append(noise_drop)
            clean_list.append(clean_f1)
            colors_list.append(color)
            config_labels.append(cfg_label)
            group_xs.append(pos)

            ci += 1
            config_counter[family] = ci
            pos += 1

        group_end = pos - 1
        model_spans.append((model_name, group_start, group_end, family))
        family_spans.setdefault(family, []).append((group_start, group_end))

        prev_family = family
        prev_model_key = model_key

    xs = np.array(xs)
    n = len(xs)

    fig, ax = plt.subplots(figsize=(11, 5.5))

    for i in range(n):
        x_c = xs[i]
        ret = retained_list[i]
        nd = drop_list[i]
        cf = clean_list[i]

        ax.bar(x_c, ret, bar_width, color=colors_list[i],
               edgecolor='white', linewidth=0.3)

        if nd > 0:
            ax.bar(x_c, nd, bar_width, bottom=ret, color='none',
                   edgecolor='#CC8888', linewidth=0.3,
                   hatch='///', facecolor='#FFEEEE')

        ax.text(x_c, ret * 0.45, f'{ret:.1f}', ha='center', va='center',
                fontsize=7, fontweight='bold', color='white')
        if nd > 0.3:
            drop_y = ret + max(nd * 0.5, 0.4)
            ax.text(x_c, drop_y, f'-{nd:.1f}', ha='center', va='center',
                    fontsize=6, fontweight='bold', color='#AA0000')
        ax.text(x_c, cf + 0.4, f'{cf:.1f}', ha='center', va='bottom',
                fontsize=6.5, fontweight='bold')

    max_f1 = max(clean_list) if clean_list else 82
    bracket_y_data = max_f1 + 3.5

    ax.set_xticks(xs)
    ax.set_xticklabels(config_labels, rotation=35, ha='right', fontsize=10)
    ax.set_ylabel('Macro F1', fontsize=13, fontweight='bold')
    ax.set_ylim(0, max_f1 + 12)
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    ax.set_xlim(xs[0] - 0.8, xs[-1] + 0.8)

    for model_name, start, end, family in model_spans:
        cmap = FAMILY_CMAP[family]
        model_list = [m[0] for m in MODELS if m[2] == family]
        n_total = len(model_list)
        try:
            ci = model_list.index(model_name)
        except ValueError:
            ci = 0
        shade = 0.38 + 0.30 * (ci / max(n_total - 1, 1))
        model_text_color = cmap(shade)

        center = (start + end) / 2.0
        ax.text(center, bracket_y_data, model_name,
                ha='center', va='bottom', fontsize=10, fontweight='bold',
                color=model_text_color, clip_on=False)

    trans = ax.get_xaxis_transform()
    for family, spans in family_spans.items():
        lo = min(s[0] for s in spans)
        hi = max(s[1] for s in spans)
        center = (lo + hi) / 2.0
        ax.text(center, -0.35, family, ha='center', va='top',
                fontsize=16, fontweight='bold', transform=trans,
                clip_on=False)

    legend_elements = [
        Patch(facecolor='#FFEEEE', edgecolor='#CC8888', hatch='///',
              label='Avg Drop due to Noise'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.01, 0.90),
              fontsize=6, frameon=True, edgecolor='gray', facecolor='white', framealpha=0.85)

    fig.tight_layout()
    plt.subplots_adjust(bottom=0.35, top=0.82)
    out_dir = os.path.join(ROOT, 'figures')
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, 'Macro_F1_Decomposed_Drop_vs_Models.pdf'),
                bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(out_dir, 'Macro_F1_Decomposed_Drop_vs_Models.png'),
                bbox_inches='tight', dpi=300)
    plt.close()
    print("Saved to plots/figures/")


chart()
