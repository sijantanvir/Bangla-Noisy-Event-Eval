import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif'],
    'font.size': 18,
    'axes.labelsize': 15,
    'axes.titlesize': 16,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'figure.dpi': 300
})

ROOT = os.path.dirname(os.path.abspath(__file__))


def get_stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return ""


def format_encoder_label(label):
    mapping = {
        'banglabert': 'BanglaBERT',
        'banglabert_large': 'BanglaBERT-L',
        'xlm-roberta-base': 'XLM-R',
        'xlm-roberta-large': 'XLM-R-L'
    }
    return mapping.get(label.lower(), label)


def plot_significance_heatmap(csv_path=None, wcr="WCR 40"):
    if csv_path is None:
        csv_path = os.path.join(ROOT, "data", "All_Models_Significance_Strict.csv")

    df = pd.read_csv(csv_path)
    df = df[df['WCR'] == wcr].copy()

    def get_direction(row, p_col, llm_col, enc_col):
        p = row[p_col]
        if p >= 0.05:
            return 0
        return 1 if row[llm_col] > row[enc_col] else -1

    df['T_Dir'] = df.apply(lambda r: get_direction(r, 'Trig Sig (p-value)', 'T_LLM_wins', 'T_BERT_wins'), axis=1)
    df['C_Dir'] = df.apply(lambda r: get_direction(r, 'Ctx Sig (p-value)', 'C_LLM_wins', 'C_BERT_wins'), axis=1)

    df['T_Stars'] = df['Trig Sig (p-value)'].apply(get_stars)
    df['C_Stars'] = df['Ctx Sig (p-value)'].apply(get_stars)
    df['T_Annot'] = df['T_Stars']
    df['C_Annot'] = df['C_Stars']

    enc_order = ['banglabert', 'banglabert_large', 'xlm-roberta-base', 'xlm-roberta-large']

    T_dir = df.pivot(index='LLM', columns='Encoder', values='T_Dir')
    T_ann = df.pivot(index='LLM', columns='Encoder', values='T_Annot')
    C_dir = df.pivot(index='LLM', columns='Encoder', values='C_Dir')
    C_ann = df.pivot(index='LLM', columns='Encoder', values='C_Annot')

    llm_order = sorted(T_dir.index.tolist())
    T_dir = T_dir.reindex(llm_order)
    T_ann = T_ann.reindex(llm_order)
    C_dir = C_dir.reindex(llm_order)
    C_ann = C_ann.reindex(llm_order)

    enc_order = [e for e in enc_order if e in T_dir.columns]
    T_dir = T_dir[enc_order]
    T_ann = T_ann[enc_order]
    C_dir = C_dir[enc_order]
    C_ann = C_ann[enc_order]

    col_rename = {e: format_encoder_label(e) for e in enc_order}
    T_dir = T_dir.rename(columns=col_rename)
    T_ann = T_ann.rename(columns=col_rename)
    C_dir = C_dir.rename(columns=col_rename)
    C_ann = C_ann.rename(columns=col_rename)

    color_enc_win = '#f27a7d'
    color_tie = '#fbf5f3'
    color_llm_win = '#26547c'

    colors = [color_enc_win, color_tie, color_llm_win]
    cmap = LinearSegmentedColormap.from_list('sig_map', colors, N=3)

    fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharey=True)

    for ax, mat, ann, title in zip(
        axes,
        [T_dir, C_dir],
        [T_ann, C_ann],
        ['Trigger Corruption', 'Context Corruption']
    ):
        sns.heatmap(
            mat,
            annot=ann,
            fmt="",
            cmap=cmap,
            vmin=-1, vmax=1, center=0,
            cbar=False,
            ax=ax,
            linewidths=2,
            linecolor='white',
            annot_kws={'fontsize': 14, 'fontweight': 'bold'}
        )
        ax.set_title(title, pad=15, fontweight='bold', fontsize=14)
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=30)

    axes[0].set_ylabel('Decoder Architecture', labelpad=10, fontsize=18, fontweight='bold')
    axes[1].set_ylabel('')
    fig.supxlabel('Encoder Architecture', x=0.50, y=0.05, fontweight='bold', fontsize=16)

    legend_elements = [
        Patch(facecolor=color_llm_win, label='LLM significantly\noutperforms Encoder'),
        Patch(facecolor=color_tie, edgecolor='#cccccc', label='No significant\ndifference'),
        Patch(facecolor=color_enc_win, label='Encoder significantly\noutperforms LLM')
    ]

    fig.legend(
        handles=legend_elements,
        loc='lower center',
        ncol=3,
        bbox_to_anchor=(0.5, -0.08),
        fontsize=14,
        frameon=False
    )

    plt.tight_layout()
    out_dir = os.path.join(ROOT, "figures")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "significance_heatmap.pdf"), format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(out_dir, "significance_heatmap.png"), format='png', bbox_inches='tight', dpi=300)
    print(f"Saved to {out_dir}/significance_heatmap.pdf")


if __name__ == "__main__":
    plot_significance_heatmap()
