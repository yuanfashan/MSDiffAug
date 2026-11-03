# -*- coding: utf-8 -*-
"""10 个化合物两两结构相似度 (Morgan Tanimoto) 热力图。

用法 (服务器): python plot_tanimoto_heatmap.py
plot_heatmap 是通用函数 (传入任意相似度矩阵即可画), 也可用于嵌入余弦相似度。
"""
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PNG = os.path.join(_HERE, 'tanimoto_heatmap_10compounds.png')
OUT_PDF = os.path.join(_HERE, 'tanimoto_heatmap_10compounds.pdf')

# SMILES 与 ms_diff_v.py / msbert.py / spec2vec_tsne.py 的 SELECTED_SMILES 一致,
# 名称与论文表 tab:compounds 的 Short Name 一致
COMPOUNDS = [
    ('O=C(OC1CC2C(O)(CC=C3CC(O)CCC32C)C4(O)CCC(O)(C(=O)C)C14C)C5=CC=C(O)C=C5', 'Qingyangshengenin'),
    ('O=C1C=C(OC=2C=C(O)C=C(O)C12)C=3C=CC=CC3', 'Chrysin'),
    ('O=C(OC)C1(O)N2C=3C=CC=CC3C4=C2C5N(CC4)CCCC5(CC)C1', 'Isovincamine'),
    ('OCC1OC(OC2C(O)C(O)C(OC2OC3CCC4(C)C(CCC5(C)C4CC(O)C6C(CCC65C)C(O)(C)CCC=C(C)C)C3(C)C)CO)C(O)C(O)C1O', 'PPD-diglucoside'),
    ('O=C(O)C12CCC(C)C(C)C2C3=CCC4C5(C)CC(O)C(O)C(C)(CO)C5CCC4(C)C3(C)CC1', 'Asiatic acid'),
    ('O=C1OC=2C(C=C1)=CC=3C=COC3C2OCC(O)C(O)(C)C', 'Oxypeucedanin hydrate'),
    ('O=C1C=2C(O)=CC(O)=CC2OC(C3=CC=C4OC(C5=CC=C(OC)C(O)=C5)C(OC4=C3)CO)C1O', 'Isosilybin'),
    ('O=C(OC1=C(OC(=O)C)C2=C(C=C1C(C)C)CCC3C2(C(=O)OC)CCCC3(C)C)C', 'Carnosic acid deriv.'),
    ('O=C1CCC2C(C)(CCC3C2(C)CCC4(C)C5CC(C)(C)CCC5(C)CCC34C)C1C', 'Friedelin'),
    ('O=C(O)C=CC1=CC=C(O)C(O)=C1', 'Caffeic acid'),
]


def tanimoto_matrix(smiles, radius=2, bits=2048):
    """两两 Morgan/Tanimoto 相似度矩阵; SMILES 解析失败的行列为 NaN。"""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator, DataStructs
    RDLogger.DisableLog("rdApp.error")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)
    fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles]
    n = len(smiles)
    mat = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i, n):
            if fps[i] is None or fps[j] is None:
                continue
            mat[i, j] = mat[j, i] = DataStructs.TanimotoSimilarity(fps[i], fps[j])
    return mat


def plot_heatmap(mat, labels, out_png=OUT_PNG, out_pdf=OUT_PDF, vmax=0.25,
                 cbar_label='Tanimoto similarity (Morgan r=2, 2048-bit)',
                 title=None):
    """通用相似度热力图: vmax 压缩色域以突出差异, 对角线 (1.0) 自然饱和。"""
    import matplotlib
    matplotlib.use("Agg")  # 服务器无显示环境
    import matplotlib.pyplot as plt

    n = len(labels)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(mat, cmap='viridis', vmin=0, vmax=vmax)
    ax.set_xticks(np.arange(n), labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(np.arange(n), labels, fontsize=9)
    for i in range(n):
        for j in range(n):
            v = mat[i, j]
            if np.isnan(v):
                continue
            txt = '1.00' if i == j else f'{v:.2f}'
            color = 'white' if v < vmax * 0.6 else 'black'
            ax.text(j, i, txt, ha='center', va='center', fontsize=8, color=color)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label(cbar_label, fontsize=9)
    if title:
        ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=600, bbox_inches='tight')
    fig.savefig(out_pdf, format='pdf', bbox_inches='tight')
    print(f'热力图已保存: {out_png}')
    return fig


def plot_tanimoto_heatmap(compounds, **kwargs):
    """compounds: [(smiles, short_name), ...] -> Tanimoto 热力图。"""
    smiles, short_names = zip(*compounds)
    mat = tanimoto_matrix(smiles)
    labels = [f'{i} {n}' for i, n in enumerate(short_names)]
    off = mat[~np.eye(len(mat), dtype=bool)]
    print(f'两两 Tanimoto: min={np.nanmin(off):.3f} median={np.nanmedian(off):.3f} '
          f'max={np.nanmax(off):.3f}')
    return plot_heatmap(mat, labels, **kwargs)


if __name__ == '__main__':
    plot_tanimoto_heatmap(
        COMPOUNDS,
        title='Pairwise structural similarity of the 10 selected compounds')
