import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 14,
    'ytick.labelsize': 10,
    'xtick.labelsize': 10,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.35,
    'grid.linestyle': '--',
})

ROOT = os.path.dirname(os.path.abspath(__file__))

FAMILY_CMAP = {
    'Gemma': plt.cm.Blues,
    'LLaMA': plt.cm.Oranges,
    'BanglaBERT': plt.cm.Reds,
    'XLM-R': plt.cm.Greens,
}

FAMILY_MODELS = {
    'Gemma': ['gemma3_1b_16bit', 'gemma3_4B_16bit'],
    'LLaMA': ['llama3_2_1b_16bit', 'llama3_1_8b_16bit_r16'],
    'BanglaBERT': ['banglabert_dsoftmax', 'banglabert_large_dsoftmax'],
    'XLM-R': ['xlm-roberta-base_dsoftmax', 'xlm-roberta-large_dsoftmax'],
}

LLM_LINES = [
    ('Gemma 3 1B IT', 'gemma3_1b_16bit', 'Gemma', 'Clean', 'Guided', 'LLM-G'),
    ('Gemma 3 1B IT', 'gemma3_1b_16bit', 'Gemma', 'Clean', 'Guideline-Free', 'LLM-GF'),
    ('Gemma 3 1B IT', 'gemma3_1b_16bit', 'Gemma', 'Combined', 'Guided', 'LLM-G'),
    ('Gemma 3 4B IT', 'gemma3_4B_16bit', 'Gemma', 'Clean', 'Guided', 'LLM-G'),
    ('Gemma 3 4B IT', 'gemma3_4B_16bit', 'Gemma', 'Clean', 'Guideline-Free', 'LLM-GF'),
    ('Gemma 3 4B IT', 'gemma3_4B_16bit', 'Gemma', 'Combined', 'Guided', 'LLM-G'),
    ('Llama 3.2 1B IT', 'llama3_2_1b_16bit', 'LLaMA', 'Clean', 'Guided', 'LLM-G'),
    ('Llama 3.2 1B IT', 'llama3_2_1b_16bit', 'LLaMA', 'Clean', 'Guideline-Free', 'LLM-GF'),
    ('Llama 3.2 1B IT', 'llama3_2_1b_16bit', 'LLaMA', 'Combined', 'Guided', 'LLM-G'),
    ('Llama 3.1 8B IT', 'llama3_1_8b_16bit_r16', 'LLaMA', 'Clean', 'Guided', 'LLM-G'),
    ('Llama 3.1 8B IT', 'llama3_1_8b_16bit_r16', 'LLaMA', 'Clean', 'Guideline-Free', 'LLM-GF'),
    ('Llama 3.1 8B IT', 'llama3_1_8b_16bit_r16', 'LLaMA', 'Combined', 'Guided', 'LLM-G'),
]

BERT_LINES = [
    ('BanglaBERT-Base', 'banglabert_dsoftmax', 'BanglaBERT', 'Clean', 'N/A', 'Encoder'),
    ('BanglaBERT-Base', 'banglabert_dsoftmax', 'BanglaBERT', 'Combined', 'N/A', 'Encoder'),
    ('BanglaBERT-Large', 'banglabert_large_dsoftmax', 'BanglaBERT', 'Clean', 'N/A', 'Encoder'),
    ('BanglaBERT-Large', 'banglabert_large_dsoftmax', 'BanglaBERT', 'Combined', 'N/A', 'Encoder'),
    ('XLM-R-Base', 'xlm-roberta-base_dsoftmax', 'XLM-R', 'Clean', 'N/A', 'Encoder'),
    ('XLM-R-Base', 'xlm-roberta-base_dsoftmax', 'XLM-R', 'Combined', 'N/A', 'Encoder'),
    ('XLM-R-Large', 'xlm-roberta-large_dsoftmax', 'XLM-R', 'Clean', 'N/A', 'Encoder'),
    ('XLM-R-Large', 'xlm-roberta-large_dsoftmax', 'XLM-R', 'Combined', 'N/A', 'Encoder'),
]

STYLE_MAP = {
    'Encoder': {'ls': '--', 'dash': (4, 1.5), 'marker': 's'},
    'LLM-G': {'ls': '-', 'dash': None, 'marker': 'o'},
    'LLM-GF': {'ls': '-.', 'dash': (1, 1.5), 'marker': 'D'},
}

BERT_STYLE_MAP = {
    'Encoder': {'ls': '-', 'dash': None, 'marker': 'o'},
}


def _alias(model):
    return 'llama3_1_8b_16bit_r16' if model == 'lamma3_1_8b_16bit_r16' else model


BERT_8_COLORS = ['#4D9CC9', '#DBA1BF', '#0072B2', '#CC79A7',
                  '#9787B9', '#4DBBA1', '#6A3D9A', '#009E73']

MODEL_HEX_COLORS = {
    'Gemma': ['#009E73', '#007250'],
    'LLaMA': ['#E69F00', '#C47A00'],
    'BanglaBERT': ['#CC79A7', '#A04B7A'],
    'XLM-R': ['#56B4E9', '#2E8BC0'],
}


def get_line_color(family, model_key, line_in_model_idx, n_lines_in_model):
    models_in_family = FAMILY_MODELS[family]
    try:
        mi = models_in_family.index(model_key)
    except ValueError:
        mi = 0
    colors = MODEL_HEX_COLORS[family]
    return colors[mi]


def get_clean_f1(df_clean, model_key, train, guide):
    guide_mask = df_clean['Guideline'].isna() if guide == 'N/A' else (df_clean['Guideline'] == guide)
    row = df_clean[
        (df_clean['Model'] == model_key) &
        (df_clean['Training_Data'] == train) &
        guide_mask
    ]
    if row.empty:
        return None
    return float(row.iloc[0]['Strict_Macro_F1'])


def get_wcr_data(df_wcr, model_key, train, guide):
    guide_mask = df_wcr['Guideline'].isna() if guide == 'N/A' else (df_wcr['Guideline'] == guide)
    rows = df_wcr[
        (df_wcr['Model'] == model_key) &
        (df_wcr['Training_Data'] == train) &
        guide_mask
    ].sort_values('WCR_Level')
    levels, f1s = [], []
    for _, r in rows.iterrows():
        levels.append(int(r['WCR_Level']))
        f1s.append(float(r['Strict_Macro_F1']))
    return levels, f1s


def draw_lines(ax, lines_list, style_map, df_clean, df_wcr, label_modifier, colors=None):
    model_line_count = {}
    for _, model_key, _, _, _, _ in lines_list:
        key = model_key
        model_line_count[key] = model_line_count.get(key, 0) + 1

    for li, (display_name, model_key, family, train, guide, setup_type) in enumerate(lines_list):
        idx = 0
        if colors is not None:
            color = colors[li]
        else:
            model_idx = {}
            for _, mk, _, _, _, _ in lines_list[:li]:
                if mk == model_key:
                    idx += 1
            color = get_line_color(family, model_key, idx, model_line_count[model_key])
        style = style_map[setup_type]

        clean_f1 = get_clean_f1(df_clean, model_key, train, guide)
        if clean_f1 is None:
            continue

        w_lvls, w_f1s = get_wcr_data(df_wcr, model_key, train, guide)

        wcr_levels = [0] + w_lvls
        f1_values = [clean_f1] + w_f1s

        label = label_modifier(display_name, train, guide, setup_type)

        plot_kwargs = dict(
            color=color,
            linestyle=style['ls'],
            marker=style['marker'],
            markersize=4,
            linewidth=1.3,
            label=label,
            zorder=3,
        )
        if style['dash'] is not None:
            plot_kwargs['dashes'] = style['dash']

        ax.plot(wcr_levels, f1_values, **plot_kwargs)


def format_ax(ax, title, ncol_legend=4, frameon=True):
    ax.set_xlabel('Word Corruption Rate (WCR) %')
    ax.set_ylabel('Macro F1')
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_xlim(-2, 42)
    ax.legend(loc='upper right',
              ncol=ncol_legend, frameon=frameon, edgecolor='gray', facecolor='white',
              framealpha=0.85, fontsize=10, handlelength=1.2,
              handletextpad=0.5, columnspacing=0.8, labelspacing=0.6, borderpad=0.3)


def plot_wcr():
    df_wcr = pd.read_csv(os.path.join(ROOT, 'data', 'WCR_Test_Results.csv'))
    df_clean = pd.read_csv(os.path.join(ROOT, 'data', 'Clean_Test_Results.csv'))
    df_wcr['Model'] = df_wcr['Model'].map(_alias)
    df_clean['Model'] = df_clean['Model'].map(_alias)

    def llm_label(display_name, train, guide, setup_type):
        suffix = ''
        if setup_type == 'LLM-G':
            suffix = ' (w/ Guide)'
        if setup_type == 'LLM-GF':
            suffix = ' (w/o Guide)'
        elif train == 'Combined':
            suffix = ' (Comb. + w/ Guide)'
        return display_name + suffix

    def bert_label(display_name, train, guide, setup_type):
        suffix = ' (Comb.)' if train == 'Combined' else ''
        return display_name + suffix

    LLM_MODEL_TRAIN_COLORS = {
        ('gemma3_1b_16bit', 'Clean'): '#0072B2',
        ('gemma3_1b_16bit', 'Combined'): '#D55E00',
        ('gemma3_4B_16bit', 'Clean'): '#009E73',
        ('gemma3_4B_16bit', 'Combined'): '#CC79A7',
        ('llama3_2_1b_16bit', 'Clean'): '#56B4E9',
        ('llama3_2_1b_16bit', 'Combined'): '#6A3D9A',
        ('llama3_1_8b_16bit_r16', 'Clean'): '#E69F00',
        ('llama3_1_8b_16bit_r16', 'Combined'): '#000000',
    }
    llm_colors_list = [LLM_MODEL_TRAIN_COLORS[(k, t)] for _, k, _, t, _, _ in LLM_LINES]

    out_dir = os.path.join(ROOT, 'figures')
    os.makedirs(out_dir, exist_ok=True)

    fig_llm, ax_llm = plt.subplots(figsize=(11.2, 5.5))
    draw_lines(ax_llm, LLM_LINES, STYLE_MAP, df_clean, df_wcr, llm_label, colors=llm_colors_list)
    ylo, yhi = ax_llm.get_ylim()
    ax_llm.set_ylim(ylo, yhi + 8)
    format_ax(ax_llm, 'LLMs', ncol_legend=3, frameon=False)
    fig_llm.savefig(os.path.join(out_dir, 'Macro_F1_vs_WCR_LLMs.pdf'), bbox_inches='tight', dpi=300)
    fig_llm.savefig(os.path.join(out_dir, 'Macro_F1_vs_WCR_LLMs.png'), bbox_inches='tight', dpi=300)
    plt.close(fig_llm)
    print(f"Saved to plots/figures/Macro_F1_vs_WCR_LLMs.png")

    fig_bert, ax_bert = plt.subplots(figsize=(11.2, 5.5))
    draw_lines(ax_bert, BERT_LINES, BERT_STYLE_MAP, df_clean, df_wcr, bert_label, colors=BERT_8_COLORS)
    ylo, yhi = ax_bert.get_ylim()
    ax_bert.set_ylim(ylo, yhi + 5)
    format_ax(ax_bert, 'BERT Models', ncol_legend=2, frameon=False)
    fig_bert.savefig(os.path.join(out_dir, 'Macro_F1_vs_WCR_BERTs.pdf'), bbox_inches='tight', dpi=300)
    fig_bert.savefig(os.path.join(out_dir, 'Macro_F1_vs_WCR_BERTs.png'), bbox_inches='tight', dpi=300)
    plt.close(fig_bert)
    print(f"Saved to plots/figures/Macro_F1_vs_WCR_BERTs.png")


if __name__ == "__main__":
    plot_wcr()
