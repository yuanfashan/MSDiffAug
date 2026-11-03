# -*- coding: utf-8 -*-
"""
案例主图绘图方法 (MSDiffAug 扩散增强案例演示图)

由 ablition6/diffusion_example.py 调用, 绘制 7 个子图:
(a) 原始谱图 -> (b) 局部稀疏采样 -> (c) m/z+强度联合嵌入(锚点作条件)
-> (d) 被选峰上的扩散帧 -> (e) softmax 映射回 token -> (f) 插回序列 -> (g) 增强谱图叠加

case 字典字段 (difussion_example.py 构造):
    smiles, precursor_mz, n_peaks,
    mz_raw, intensity_raw,                    # 原始峰 (未取整)
    input_ids (100,), intensity_norm (100,),  # token 序列与归一化强度
    selected_pos (k,), anchor_pos,            # 位置 (0=前体, 1..n_peaks=峰)
    selected_mz (k,),                         # 被选峰 m/z (token 换算)
    T, t_used,                                # 扩散总步数与增强所用步
    emb_stages: OrderedDict 标签 -> (k, D)    # 各阶段被选峰嵌入向量
    softmax: {'mz_axis': (100002,), 'probs': (k, 100002)}
    softmax_info: [dict] 每被选峰: pos, true_mz, true_token,
                   top10_tokens, top10_mz, top10_probs,
                   sampled, sampled_mz, sampled_prob, delta_mz
    aug_seq (100,), hidden_size
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# 配色
C_ORIG = '#8A97A8'      # 原始谱图灰蓝
C_ANCHOR = '#2E5FA3'    # 锚点蓝 (条件/保留)
C_SEL = '#D64541'       # 被选红 (被扰动)
C_AUG1 = '#2E8B57'      # 增强视图 1 绿
C_AUG2 = '#E67E22'      # 增强视图 2 橙
C_PAD = '#EFEFEF'       # PAD 灰
C_PREC = '#5D5D5D'      # 前体深灰


def token_to_mz(tokens):
    """token -> m/z: (token-2)/100"""
    return (np.asarray(tokens) - 2) / 100.0


def _stem(ax, x, y, color, lw=1.0, ms=3.0, marker='o', alpha=1.0, zorder=2):
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    ax.vlines(x, 0, y, colors=color, linewidths=lw, alpha=alpha, zorder=zorder)
    ax.plot(x, y, linestyle='none', marker=marker, markersize=ms,
            color=color, alpha=alpha, zorder=zorder)


def _handle(color, label, marker='o', ls='-', ms=5):
    return Line2D([], [], color=color, marker=marker, linestyle=ls,
                  label=label, markersize=ms)


def plot_original_spectrum(ax, case):
    """(a) 原始谱图"""
    mz, y = case['mz_raw'], case['intensity_raw']
    _stem(ax, mz, y, C_ORIG, lw=1.0, ms=2.5)
    ax.set_xlabel('m/z (Da)')
    ax.set_ylabel('Intensity')
    ax.set_title('(a) Original spectrum', fontsize=10)
    ax.set_xlim(mz.min() - 10, mz.max() + 10)
    ax.set_ylim(0, y.max() * 1.1)
    ax.text(0.02, 0.97, f"{case['n_peaks']} fragment peaks, precursor "
                        f"{case['precursor_mz']:.2f} Da",
            transform=ax.transAxes, va='top', fontsize=8)


def plot_sparse_sampling(ax, case):
    """(b) 局部稀疏采样: 被选峰标红, 其余为锚点"""
    mz, y = case['mz_raw'], case['intensity_raw']
    sel_raw = np.asarray(case['selected_pos'] - 1)
    anchors = np.array([i for i in range(case['n_peaks']) if i not in sel_raw])
    _stem(ax, mz, y, '#C9D1DC', lw=0.7, ms=1.5)
    _stem(ax, mz[anchors], y[anchors], C_ANCHOR, lw=1.1, ms=3.0)
    _stem(ax, mz[sel_raw], y[sel_raw], C_SEL, lw=1.8, ms=6.0, marker='*', zorder=3)
    for s in sel_raw:
        ax.annotate(f"{mz[s]:.2f}", (mz[s], y[s]), textcoords='offset points',
                    xytext=(0, 6), ha='center', fontsize=8, color=C_SEL)
    ax.set_xlabel('m/z (Da)')
    ax.set_ylabel('Intensity')
    ax.set_title('(b) Localized sparse sampling', fontsize=10)
    ax.set_xlim(mz.min() - 10, mz.max() + 10)
    ax.set_ylim(0, y.max() * 1.1)
    ax.legend(handles=[_handle(C_ANCHOR, 'anchors (context)', 'o'),
                       _handle(C_SEL, 'selected (corrupted)', '*')],
              fontsize=8, loc='upper right')


def plot_joint_embedding(ax, case):
    """(c) m/z+强度联合嵌入示意: h_i = E_mz(z_i) + f_I(I_i), 锚点作为去噪条件"""
    rel = case['intensity_raw'] / max(case['intensity_raw'])
    sel_raw = sorted(case['selected_pos'] - 1)
    anchor_raw = [i for i in range(case['n_peaks']) if i not in sel_raw]
    anchor_top = sorted(anchor_raw, key=lambda i: -rel[i])[:2]
    rows = ([{'raw': i, 'role': 'anchor'} for i in anchor_top] +
            [{'raw': i, 'role': 'selected'} for i in sel_raw[:2]])
    top = 0.87
    row_h = 0.115
    gap = 0.05
    ys = [(top - r * (row_h + gap), row_h) for r in range(len(rows))]

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('(c) Input of the diffusion model',
                 fontsize=10)

    for row, (y0, h) in zip(rows, ys):
        base_y = y0 + h / 2
        i = row['raw']
        role = row['role']
        mz = case['mz_raw'][i]
        pc = C_ANCHOR if role == 'anchor' else C_SEL
        inten = case['intensity_raw'][i]
        istr = (f"{inten:.0f}" if inten >= 100 else
                f"{inten:.1f}" if inten >= 10 else f"{inten:.2f}")
        ax.text(0.01, base_y, f"({mz:.2f}, {istr})", fontsize=7,
                ha='left', va='center', color=pc)
        # token 嵌入条 (m/z)
        bar_h = 0.045
        ax.add_patch(Rectangle((0.16, base_y - bar_h / 2), 0.11, bar_h,
                               facecolor='#5B8BD4', edgecolor='none'))
        ax.text(0.215, base_y + 0.035, r'$E_{mz}(z)$', fontsize=7,
                ha='center', color='#3A5B8C')
        # 加号位于 E_mz 与 f_I 之间: E_mz + f_I
        ax.text(0.286, base_y, '+', fontsize=9, ha='center', va='center')
        # 强度嵌入条 (宽度随相对强度)
        w_int = 0.02 + 0.10 * rel[i]
        ax.add_patch(Rectangle((0.30, base_y - bar_h / 2), w_int, bar_h,
                               facecolor='#F0A35E', edgecolor='none'))
        ax.text(0.30 + w_int / 2, base_y + 0.035, r'$f(I)$', fontsize=7,
                ha='center', color='#B06A2B')
        # f_I 指向 h_i 的箭头
        ax.add_patch(FancyArrowPatch((0.30 + w_int, base_y), (0.452, base_y),
                                     arrowstyle='-|>', mutation_scale=9,
                                     color='#3D4C63', lw=1.2))
        # 联合向量 h_i
        ax.add_patch(Rectangle((0.46, base_y - bar_h / 2), 0.15, bar_h,
                               facecolor='#3D4C63', edgecolor='none'))
        ax.text(0.535, base_y + 0.035, r'$h$', fontsize=8, ha='center', color='#3D4C63')
        # 锚点 -> 上下文; 被选 -> 加噪 -> 去噪器
        if role == 'anchor':
            ax.add_patch(FancyArrowPatch((0.615, base_y), (0.70, base_y),
                                         arrowstyle='-|>', mutation_scale=9,
                                         color=C_ANCHOR, lw=1.0))
        else:
            ax.add_patch(FancyArrowPatch((0.615, base_y), (0.715, base_y),
                                         arrowstyle='-|>', mutation_scale=9,
                                         color=C_SEL, lw=1.0))
            ax.text(0.762, base_y, r'$\oplus\,\sigma_t\varepsilon$', fontsize=8,
                    ha='center', va='center', color=C_SEL)
            ax.add_patch(FancyArrowPatch((0.81, base_y), (0.86, base_y),
                                         arrowstyle='-|>', mutation_scale=9,
                                         color=C_SEL, lw=1.0))
    # 上下文框 (覆盖锚点行)
    anchor_bands = [(y0, h) for row, (y0, h) in zip(rows, ys)
                    if row.get('role') == 'anchor']
    if anchor_bands:
        ymin = min(y0 for y0, _ in anchor_bands) - 0.05
        ymax = max(y0 + h for y0, h in anchor_bands) + 0.05
        ax.add_patch(Rectangle((0.70, ymin), 0.12, ymax - ymin,
                               facecolor='#DCE8F8', edgecolor=C_ANCHOR, lw=1.2))
        ax.text(0.76, (ymin + ymax) / 2, 'context(anchors)', fontsize=7.5,
                ha='center', va='center', color=C_ANCHOR)
        cy = (ymin + ymax) / 2
        ax.add_patch(FancyArrowPatch((0.825, cy), (0.858, cy), arrowstyle='-|>',
                                     mutation_scale=9, color=C_ANCHOR, lw=1.2))
        ax.text(0.842, cy + 0.045, 'cond.', fontsize=7, ha='center', color=C_ANCHOR)
    # 去噪器
    ymin_all = min(y0 for y0, _ in ys) - 0.05
    ymax_all = max(y0 + h for y0, h in ys) + 0.05
    ax.add_patch(Rectangle((0.86, ymin_all), 0.125, ymax_all - ymin_all,
                           facecolor='#2E5FA3', edgecolor='none'))
    ax.text(0.9225, (ymin_all + ymax_all) / 2, 'Sequence\nDiffusion',
            fontsize=8, ha='center', va='center', color='white')


def plot_diffusion_frames(ax, case):
    """(d) 被选峰嵌入沿扩散轨迹的状态: 前向 0->500->999, 反向 999->500->0 (锚点保持干净)"""
    stages = case['emb_stages']
    labels = list(stages.keys())
    k, D = stages[labels[0]].shape
    img = np.concatenate([stages[l] for l in labels], axis=1)
    allv = np.concatenate([stages[l].ravel() for l in labels])
    v = np.percentile(np.abs(allv), 98)
    im = ax.imshow(img, aspect='auto', cmap='RdBu_r', vmin=-v, vmax=v,
                   interpolation='nearest')
    for i in range(1, len(labels)):
        ax.axvline(i * D - 0.5, color='k', lw=1.0)
    ax.set_yticks(range(k))
    ax.set_yticklabels([f"{mz:.2f} Da" for mz in case['selected_mz']], fontsize=8)
    ax.set_xticks([(i + 0.5) * D for i in range(len(labels))])
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_title('(d) Diffusion on selected peaks', fontsize=10)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label('embedding value', fontsize=8)
    cb.ax.tick_params(labelsize=7)


def plot_softmax_mapping(axes, case):
    """(e) softmax 概率分布: 去噪后的嵌入映射回离散 m/z token"""
    probs = case['softmax']['probs']
    mz_axis = case['softmax']['mz_axis']
    infos = case['softmax_info']
    for r, (axr, info) in enumerate(zip(axes, infos)):
        lo, hi = info['true_mz'] - 25.0, info['true_mz'] + 25.0
        sel = (mz_axis >= lo) & (mz_axis <= hi)
        axr.fill_between(mz_axis[sel], probs[r][sel], color='#9FB6D4', lw=0)
        axr.axvline(info['true_mz'], color=C_SEL, ls='--', lw=1.2)
        in_win = [i for i in range(len(info['top10_mz'])) if lo <= info['top10_mz'][i] <= hi]
        axr.plot([info['top10_mz'][i] for i in in_win],
                 [info['top10_probs'][i] for i in in_win],
                 linestyle='none', marker='o', ms=3.5, color=C_ANCHOR, zorder=3)
        axr.plot([info['sampled_mz']], [info['sampled_prob']],
                 linestyle='none', marker='*', ms=11, color=C_AUG1, zorder=4)
        axr.text(0.98, 0.9, f"true {info['true_mz']:.2f}  →  {info['sampled_mz']:.2f}",
                 transform=axr.transAxes, ha='right', va='top', fontsize=7.5)
        axr.set_ylabel('P(token)', fontsize=8)
        axr.set_ylim(0, max(probs[r][sel].max(), 1e-12) * 1.2)
        axr.set_xlim(lo, hi)
        axr.tick_params(labelsize=7)
        if r == 0:
            axr.set_title('(e) Softmax over the m/z vocabulary — map back to discrete tokens',
                          fontsize=10)
            axr.legend(handles=[_handle(C_ANCHOR, 'top-10 candidates', 'o'),
                                _handle(C_SEL, 'true m/z', 'o', ls='--'),
                                _handle(C_AUG1, 'reconstructed m/z', '*')],
                       fontsize=7, loc='upper left')
        if r == len(axes) - 1:
            axr.set_xlabel('m/z (Da)')
        else:
            axr.set_xticklabels([])
    if len(infos) > len(axes):
        axes[-1].text(0.02, 0.5, f"+ {len(infos) - len(axes)} more selected peaks ...",
                      transform=axes[-1].transAxes, fontsize=7.5, va='center')
    for axr in axes[len(infos):]:
        axr.axis('off')


def plot_sequence_insert(ax, case, fig):
    """(f) 采样得到的 token 插回序列 (仅被选位置改变)"""
    ids = case['input_ids']
    a1 = case['aug_seq']
    inten = case['intensity_norm']
    L = case['n_peaks'] + 3
    cmap = plt.get_cmap('viridis')
    norm = Normalize(0, 1000)
    rows = [('Original $x$', ids, 0), ('Augmented $x_1$', a1, 1)]
    for r, (_, seq, _) in enumerate(rows):
        for p in range(L):
            t = int(seq[p])
            if t >= 2:
                color = cmap(norm((t - 2) / 100.0))
                alpha = 0.30 + 0.70 * float(inten[p])
            elif t == 0:
                color, alpha = C_PAD, 1.0
            else:
                color, alpha = '#F5D76E', 1.0
            if p == 0:
                color, alpha = C_PREC, 1.0
            ax.add_patch(Rectangle((p, 1 - r), 1, 1, facecolor=color, alpha=alpha,
                                   edgecolor='none'))
            if p in case['selected_pos']:
                ax.add_patch(Rectangle((p, 1 - r), 1, 1, facecolor='none',
                                       edgecolor=C_SEL if r == 0 else C_AUG1, lw=1.8))
    ax.set_xlim(0, L)
    ax.set_ylim(0, 2)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels([rows[1][0], rows[0][0]], fontsize=8)
    ax.set_xticks(range(0, L, 5))
    ax.set_xticklabels([str(p) for p in range(0, L, 5)], fontsize=7)
    ax.set_xlabel('position (0 = precursor)')
    ax.set_title('(f) Insert sampled tokens back into the sequence', fontsize=10)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label('m/z (Da)', fontsize=8)
    cb.ax.tick_params(labelsize=7)


def plot_augmented_spectrum(ax, case):
    """(g) 增强谱图: 锚点保留, 被选位置替换为采样 token"""
    orig_mz = token_to_mz(case['input_ids'][1:case['n_peaks'] + 1])
    y = case['intensity_raw']
    sel_raw = np.asarray(case['selected_pos'] - 1)
    anchors = np.array([i for i in range(case['n_peaks']) if i not in sel_raw])
    aug_mz = token_to_mz(case['aug_seq'][case['selected_pos']])
    _stem(ax, orig_mz, y, '#C9D1DC', lw=0.7, ms=1.5)
    _stem(ax, orig_mz[anchors], y[anchors], C_ANCHOR, lw=1.2, ms=3.0)
    _stem(ax, orig_mz[sel_raw], y[sel_raw], C_SEL, lw=1.0, ms=4.0, marker='x', zorder=3)
    _stem(ax, aug_mz, y[sel_raw], C_AUG1, lw=1.8, ms=6.0, marker='*', zorder=4)
    lines = [f"{orig_mz[s]:.2f} → {m:.2f} (Δ{m - orig_mz[s]:+.2f})"
             for s, m in zip(sel_raw, aug_mz)]
    ax.text(0.02, 0.97, 'selected peaks:\n' + '\n'.join(lines),
            transform=ax.transAxes, va='top', fontsize=7.5,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#CCCCCC', alpha=0.85))
    ax.set_xlabel('m/z (Da)')
    ax.set_ylabel('Intensity')
    ax.set_title('(g) Augmented spectrum', fontsize=10)
    ax.set_xlim(orig_mz.min() - 10, orig_mz.max() + 10)
    ax.set_ylim(0, y.max() * 1.1)
    ax.legend(handles=[_handle('#C9D1DC', 'original peaks'),
                       _handle(C_ANCHOR, 'preserved (anchors)'),
                       _handle(C_SEL, 'corrupted positions', 'x'),
                       _handle(C_AUG1, 'augmented peaks', '*')],
              fontsize=7, loc='upper right')


def make_case_figure(case, save_path, figsize=(12.5, 16.0), dpi=600):
    """组装 (a)-(g) 并保存 png + pdf"""
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(6, 2, height_ratios=[1.1, 1.0, 0.85, 1.35, 0.5, 1.2],
                          hspace=0.62, wspace=0.28,
                          left=0.055, right=0.975, top=0.945, bottom=0.045)
    ax_a = fig.add_subplot(gs[0, 0])
    plot_original_spectrum(ax_a, case)
    ax_b = fig.add_subplot(gs[0, 1])
    plot_sparse_sampling(ax_b, case)
    ax_c = fig.add_subplot(gs[1, :])
    plot_joint_embedding(ax_c, case)
    ax_d = fig.add_subplot(gs[2, :])
    plot_diffusion_frames(ax_d, case)
    ax_e = fig.add_subplot(gs[3, :])
    ax_e.axis('off')
    k = min(len(case['softmax_info']), 4)
    inner = GridSpecFromSubplotSpec(k, 1, subplot_spec=ax_e, hspace=0.5)
    axes_e = [fig.add_subplot(s) for s in inner]
    plot_softmax_mapping(axes_e, case)
    ax_f = fig.add_subplot(gs[4, :])
    plot_sequence_insert(ax_f, case, fig)
    ax_g = fig.add_subplot(gs[5, :])
    plot_augmented_spectrum(ax_g, case)

    fig.savefig(save_path + '.png', dpi=dpi)
    fig.savefig(save_path + '.pdf')
    plt.close(fig)
    print(f"figure saved: {save_path}.png / {save_path}.pdf")
