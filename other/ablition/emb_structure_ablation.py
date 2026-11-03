# -*- coding: utf-8 -*-
"""MSDiffAug 嵌入结构感知消融对比 (structure-awareness ablation)。

同一组化合物对 (全库普查, 每 T 箱均匀随机保留 ≤500 化合物对, 同化合物
对的多个谱对取余弦中位数聚合) 上, 对比各消融变体的嵌入余弦结构感知
能力。化合物对只采样一次并在所有模型间共享 (匹配对比)。每模型输出:
  AUC (analog T≥0.5 vs unrelated T<0.3)、平衡 Spearman、
  各 T 箱余弦中位数。

输出 (PDF/PNG):
  a) 结构感知阶梯: 每 T 箱中位数余弦 (含分箱谱余弦"谱重叠"基线)
  b) 结构感知 AUC 柱状图
另存逐对 CSV 与控制台统计表。

用法: python plot/emb_structure_ablation.py
"""
import csv
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _REPO_ROOT,
           os.path.join(_REPO_ROOT, "ablition", "NOCL"),
           os.path.join(_REPO_ROOT, "ablition", "NOMP"),
           os.path.join(_REPO_ROOT, "ablition5", "msbert"),
           os.path.join(_REPO_ROOT, "ablition8")):
    if _p not in sys.path:
        sys.path.append(_p)

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # 服务器无显示环境
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, wilcoxon
from tqdm import tqdm

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import DataStructs, rdFingerprintGenerator

# from ablition8.msdiffaug import MSDiffAug
from ablition2.msdiffaug import MSDiffAug_MaskTop5PlusRandom
from ablition2.msdiffaug import MSDiffAug_MaskTop5PlusRandom
from ablition.NOCL.NOCL import MSMaskPredict
from ablition.NOMP.NOMP import MSDiffCL
from ablition5.msbert.msbert import MSBERT_MaskTop5PlusRandom
from data_process.process_data import MakeTrainData
from utils.ms_utils import ParseOrbitrap, ModelEmbed

# ---------------------------------------------------------------- 路径/参数 --
# (名称, 模型类, 构造参数, 权重候选[绝对路径, 按序取第一个存在的])
MODELS = [
    # dict(name="Random init", cls=MSDiffAug_MaskTop5PlusRandom, args=(100002, 512, 6, 16, 0, 100, 3),
    #      weights=[], color="#999999"),
    # dict(name="NO CL", cls=MSMaskPredict, args=(100002, 512, 6, 16, 0, 100, 3),
    #      weights=["/home/ausu/projects/yfs/MSDiffAugServer/ablition/pth/nocl/MSNOCL_23.pt"],
    #      color="#1f77b4"),
    # dict(name="NO MP", cls=MSDiffCL, args=(100002, 512, 6, 16, 0.1, 100, 3),
    #      weights=["/home/ausu/projects/yfs/MSDiffAugServer/ablition/pth/nomp/MSDiffCL_1.pt"],
    #      color="#ff7f0e"),
    dict(name="Masking Aug", cls=MSBERT_MaskTop5PlusRandom, args=(100002, 512, 6, 16, 0, 100, 3),
         weights=["/home/ausu/projects/yfs/MSDiffAugServer/ablition5/msbert/msbert_random/pth/msbert_14.pt",],
         color="#8ca02c"),
    dict(name="Random Perturbation Aug", cls=MSBERT_MaskTop5PlusRandom, args=(100002, 512, 6, 16, 0, 100, 3),
         weights=["/home/ausu/projects/yfs/MSDiffAugServer/ablition2/pth/MaskTop5PlusRandom_traditional_all/MSDiffAug_38.pt", ],
         color="#2ca02c"),
    dict(name="VAE Aug", cls=MSDiffAug_MaskTop5PlusRandom, args=(100002, 512, 6, 16, 0, 100, 3),
         weights=["/home/ausu/projects/yfs/MSDiffAugServer/vae_baseline/pth/MaskTop5PlusRandom_vae/MSDiffAug_44.pt"],
         color="#1fae90"),
    dict(name="full MSDiffAug", cls=MSDiffAug_MaskTop5PlusRandom, args=(100002, 512, 6, 16, 0, 100, 3),
         weights=["/home/ausu/projects/yfs/MSDiffAugServer/ablition8/p/t100/MSDiffAug_126.pt"],
         color="#d62728"),

]

_DATA_CANDIDATES = [
    "/home/ausu/projects/yfs/data/GNPSdata_f2/ob_train_ref.pickle",
    os.path.join(_REPO_ROOT, "data/GNPS/test_ref.pickle"),
]

MAX_SPEC_PER_COMPOUND = 5   # 每化合物保留的最多谱图数
N_BIN = 500                 # 每 T 箱均匀随机保留的最大化合物对数 (挖掘与报告均用)
SEED = 42
BATCH_SIZE = 64             # 嵌入计算批大小
FP_RADIUS, FP_BITS = 2, 2048

_T_REL = 0.3                # unrelated 带上界 / related 带下界
_T_ANA = 0.5                # analog 带下界

BIN_EDGES = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
             (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9),
             (0.9, 0.999999)]
BIN_LABELS = ['[0,0.1)', '[0.1,0.2)', '[0.2,0.3)', '[0.3,0.4)', '[0.4,0.5)',
              '[0.5,0.6)', '[0.6,0.7)', '[0.7,0.8)', '[0.8,0.9)',
              '[0.9,1.0)']


def _first_existing(candidates, what):
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{what} 不存在, 检查: {candidates}")


def _rfmt(v):
    return "-" if (v is None or np.isnan(v)) else f"{v:.3f}"


def _spearman(x, y, min_n=20):
    if len(x) < min_n or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan
    rho, p = spearmanr(x, y)
    return float(rho), float(p)


def _auc(x_neg, x_pos):
    """嵌入余弦区分正类 (类似物) 与负类 (无关分子) 的 ROC AUC (秩和形式)。"""
    if len(x_neg) == 0 or len(x_pos) == 0:
        return np.nan
    n1, n2 = len(x_pos), len(x_neg)
    ranks = np.argsort(np.argsort(np.concatenate([x_pos, x_neg]))).astype(np.float64) + 1
    r1 = ranks[:n1].sum()
    return float((r1 - n1 * (n1 + 1) / 2) / (n1 * n2))


def _mine_pairs(comp_fps, rng, keep):
    """全库普查 + 蓄水池抽样: 遍历全部化合物对 (a<b), 对每个 T 箱精确
    均匀地随机保留至多 keep 对 (池不足则全量)。所有窗口 hi < 1.0,
    同化合物对 (T=1.0) 天然不落入。返回 {(i, j): tanimoto}, i/j 为化合物索引。"""
    m = len(comp_fps)
    pools = [np.empty((0, 2), dtype=np.int64) for _ in BIN_EDGES]
    for a in tqdm(range(m - 1), desc="Census + reservoir per T bin"):
        t = np.asarray(DataStructs.BulkTanimotoSimilarity(comp_fps[a],
                                                          comp_fps[a + 1:]))
        for w, (lo, hi) in enumerate(BIN_EDGES):
            cand = np.where((t >= lo) & (t < hi))[0]
            if len(cand) == 0:
                continue
            s = min(len(cand), keep)
            b_new = np.column_stack([np.full(s, a, dtype=np.int64),
                                     a + 1 + rng.choice(cand, s, replace=False)])
            pool = np.vstack([pools[w], b_new])
            if len(pool) > keep:
                pool = pool[rng.choice(len(pool), keep, replace=False)]
            pools[w] = pool
    mined = {}
    for pool in pools:
        for a, b in pool:
            mined[(int(a), int(b))] = float(
                DataStructs.TanimotoSimilarity(comp_fps[a], comp_fps[b]))
    return mined


def main():
    RDLogger.DisableLog('rdApp.error')  # 数据中混有非 SMILES 编号, 屏蔽 RDKit 报错
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    data_file = _first_existing(_DATA_CANDIDATES, "数据 pickle")

    # ---- 数据: 每化合物至多 MAX_SPEC_PER_COMPOUND 张谱 (保留同化合物多谱) ----
    _, msms, precursor, smiles = ParseOrbitrap(data_file)
    keep, seen_count = [], {}
    for k, s in enumerate(smiles):
        if not s:
            continue
        c = seen_count.get(s, 0)
        if c < MAX_SPEC_PER_COMPOUND:
            seen_count[s] = c + 1
            keep.append(k)
    msms = [msms[k] for k in keep]
    precursor = [precursor[k] for k in keep]
    smiles = [smiles[k] for k in keep]
    print(f"谱图数 (每化合物 ≤ {MAX_SPEC_PER_COMPOUND}): {len(smiles)}")

    # ---- 结构指纹 (按 smiles 缓存, 仅保留可解析结构) ----
    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS,
                                                       fpSize=FP_BITS)
    fp_cache, ok_idx = {}, []
    for k, s in enumerate(smiles):
        if s not in fp_cache:
            mol = Chem.MolFromSmiles(s)
            fp_cache[s] = fp_gen.GetFingerprint(mol) if mol is not None else None
        if fp_cache[s] is not None:
            ok_idx.append(k)
    msms = [msms[k] for k in ok_idx]
    precursor = [precursor[k] for k in ok_idx]
    smiles = [smiles[k] for k in ok_idx]
    print(f"可解析结构谱图: {len(smiles)}")

    comp_of, comp_ids = {}, []
    for s in smiles:
        if s not in comp_of:
            comp_of[s] = len(comp_of)
        comp_ids.append(comp_of[s])
    comp_ids = np.array(comp_ids)
    print(f"唯一化合物: {len(comp_of)}")

    # ---- 化合物级结构: 每化合物谱索引分组 + 单一指纹 (消除谱对伪重复的聚合基础) ----
    spec_of = {}
    for i, c in enumerate(comp_ids):
        spec_of.setdefault(int(c), []).append(i)
    comp_smiles = [None] * len(comp_of)
    for s, c in comp_of.items():
        comp_smiles[c] = s
    comp_fps = [fp_cache[s] for s in comp_smiles]

    data, _ = MakeTrainData(msms, precursor, maxlen=100)

    # ---- 化合物对只采样一次, 所有模型共享 (匹配对比) ----
    rng = np.random.default_rng(SEED)

    mined = _mine_pairs(comp_fps, rng, N_BIN)
    pairs_arr = np.array(sorted(mined), dtype=np.int64).reshape(-1, 2)
    tan_all = np.array([mined[(int(a), int(b))] for a, b in pairs_arr])
    print(f"化合物对: mined {len(pairs_arr)}")

    mask_unrel = tan_all < _T_REL
    mask_ana = tan_all >= _T_ANA
    neg_idx = np.where(mask_unrel)[0]
    pos_idx = np.where(mask_ana)[0]
    bin_idx = [np.where((tan_all >= lo) & (tan_all < hi))[0]
               for lo, hi in BIN_EDGES]
    bin_bal = [rng.choice(w, min(len(w), N_BIN), replace=False)
               for w in bin_idx]
    bal_idx = np.concatenate(bin_bal)

    def _summarize(cos_all, name, color):
        auc = _auc(cos_all[neg_idx], cos_all[pos_idx])
        rho_bal, _ = _spearman(cos_all[bal_idx], tan_all[bal_idx])
        unrel_med = float(np.median(cos_all[mask_unrel]))
        ana_med = float(np.median(cos_all[mask_ana]))
        bin_meds = [float(np.median(cos_all[b])) if len(b) else np.nan
                    for b in bin_bal]
        return dict(name=name, auc=auc, rho_bal=rho_bal,
                    unrel=unrel_med, ana=ana_med,
                    bin_meds=bin_meds, color=color, cos_all=cos_all)

    # ---- 谱重叠基线: 0.01 Da 分箱谱余弦 (token×强度向量, 不经过模型) ----
    vs, ns = [], []
    for ids, inten, _, _ in data:
        v = {}
        for tok, it in zip(ids, np.asarray(inten, dtype=np.float32).ravel()):
            if it > 0:
                v[int(tok)] = float(it)
        vs.append(v)
        ns.append(float(np.sqrt(sum(x * x for x in v.values()))) or 1e-12)

    def _cos_pair(a, b):
        va, vb = vs[a], vs[b]
        if len(va) > len(vb):
            va, vb = vb, va
        s = sum(i * vb.get(t, 0.0) for t, i in va.items())
        return s / (ns[a] * ns[b])

    def _pair_cos_base(ca, cb):
        vals = [_cos_pair(i, j) for i in spec_of[ca] for j in spec_of[cb]]
        return float(np.median(vals))

    cos_base = np.clip(np.array([_pair_cos_base(int(a), int(b))
                                 for a, b in pairs_arr]), -1.0, 1.0)
    results = [_summarize(cos_base, "Spectral overlap", "#999999")]

    # ---- 逐模型: 嵌入 -> 同一组谱对的余弦 -> 指标 ----
    for spec in MODELS:
        name = spec["name"]
        model = spec["cls"](*spec["args"])
        if spec["weights"]:
            wf = _first_existing(spec["weights"], f"{name} 权重")
            model.load_state_dict(torch.load(wf, map_location="cpu"))
            print(f"\n[{name}] 权重: {wf}")
        else:
            # 随机初始化基线: 清零 intensity_linear。否则随机方向 w 经强度池化
            # 把所有谱对齐到同一方向, 余弦被强度分布伪影抬高 (无关对 ≈0.66)
            model.intensity_linear.weight.data.zero_()
            if model.intensity_linear.bias is not None:
                model.intensity_linear.bias.data.zero_()
            print(f"\n[{name}] 随机初始化 (未加载权重, intensity_linear 已清零)")
        emb = ModelEmbed(model, data, BATCH_SIZE)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)

        def _pair_cos_emb(ca, cb):
            # 同化合物对的多个谱对 (≤25) 取余弦中位数, 消除谱级伪重复
            return float(np.median(np.clip(
                emb[spec_of[ca]] @ emb[spec_of[cb]].T, -1.0, 1.0)))

        cos_m = np.array([_pair_cos_emb(int(a), int(b))
                          for a, b in pairs_arr])
        d = _summarize(cos_m, name, spec["color"])
        results.append(d)

    # ---- 显著性: AUC 配对 bootstrap 95% CI + 阶梯图逐箱 Wilcoxon ----
    boot_rng = np.random.default_rng(SEED + 1)
    n_boot = 2000
    auc_boot = {r["name"]: np.empty(n_boot) for r in results}
    for b in range(n_boot):
        pi = boot_rng.choice(pos_idx, len(pos_idx), replace=True)
        ni = boot_rng.choice(neg_idx, len(neg_idx), replace=True)
        for r in results:
            auc_boot[r["name"]][b] = _auc(r["cos_all"][ni],
                                          r["cos_all"][pi])
    for r in results:
        lo, hi = np.percentile(auc_boot[r["name"]], [2.5, 97.5])
        r["auc_ci"] = (r["auc"] - lo, hi - r["auc"])

    def _stars(p):
        return ("***" if p < 0.001 else "**" if p < 0.01
                else "*" if p < 0.05 else "ns")

    by_name = {r["name"]: r for r in results}
    msd = by_name["full MSDiffAug"]
    aug2 = by_name["Masking Aug"]
    aug1 = by_name["Random Perturbation Aug"]
    # 配对 bootstrap p (同组重采样, 双侧): MSDiffAug vs 传统增强
    for other in (aug2, aug1):
        p1 = float(np.mean(auc_boot[msd["name"]]
                           <= auc_boot[other["name"]]))
        other["p_auc"] = min(1.0, 2.0 * min(p1, 1.0 - p1))
    # 逐箱 Wilcoxon signed-rank (MSDiffAug vs Aug2, 同组 500 对配对)
    bin_p = []
    for idx in bin_bal:
        d = msd["cos_all"][idx] - aug2["cos_all"][idx]
        d = d[d != 0]
        bin_p.append(float(wilcoxon(d).pvalue) if len(d) else 1.0)
    # 各模型逐箱 bootstrap 中位数 95% CI (每箱同一组重采样索引, 跨模型配对)
    bin_cis = {r["name"]: [] for r in results}
    for idx in bin_bal:
        meds_boot = {r["name"]: np.empty(n_boot) for r in results}
        for b in range(n_boot):
            ri = boot_rng.choice(idx, len(idx), replace=True)
            for r in results:
                meds_boot[r["name"]][b] = np.median(r["cos_all"][ri])
        for r in results:
            lo, hi = np.percentile(meds_boot[r["name"]], [2.5, 97.5])
            m = float(np.median(r["cos_all"][idx]))
            bin_cis[r["name"]].append((m - lo, hi - m))
    print("\nAUC bootstrap 95% CI 与配对 p (双侧, vs full MSDiffAug):")
    for r in results:
        lo = r["auc"] - r["auc_ci"][0]
        hi = r["auc"] + r["auc_ci"][1]
        extra = (f"  p={r['p_auc']:.4f} ({_stars(r['p_auc'])})"
                 if "p_auc" in r else "")
        print(f"  {r['name']:<18} {r['auc']:.3f} [{lo:.3f}, {hi:.3f}]{extra}")
    print("逐箱 Wilcoxon p (MSDiffAug vs Aug2): "
          + " ".join(f"{p:.1e}" for p in bin_p))

    # ---- 报告 ----
    print(f"\n========== 消融对比汇总 (同一组 {len(tan_all)} 化合物对) ==========")
    print(f"{'模型':<18}{'AUC':>8}{'bal ρ':>8}{'无关中位':>10}{'analog 中位':>12}")
    for r in results:
        print(f"{r['name']:<18}{r['auc']:>8.3f}{_rfmt(r['rho_bal']):>8}"
              f"{_rfmt(r['unrel']):>10}{_rfmt(r['ana']):>12}")

    print(f"\n各 T 箱中位数余弦矩阵 (行=箱, 列=模型, 末列=箱内化合物对数; "
          f"每箱均匀随机保留 ≤{N_BIN} 化合物对):")
    print(f"{'T 区间':<12}" + "".join(f"{r['name']:>12}" for r in results) + f"{'N':>8}")
    for k, lab in enumerate(BIN_LABELS):
        row = "".join(f"{r['bin_meds'][k]:>12.3f}" if not np.isnan(r['bin_meds'][k])
                      else f"{'-':>12}" for r in results)
        print(f"{lab:<12}{row}{len(bin_idx[k]):>8}")

    # ---- 图 ----
    plt.rcParams.update({'font.family': 'sans-serif',
                         'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
                         'font.size': 12})
    fig, ax = plt.subplots(figsize=(7.5, 4.6))

    # 结构感知阶梯 (每 T 箱中位数余弦, 全部为不同化合物的谱对)
    stair_colors = ["#006493", "#9763ad", "#f15f75", "#ffd680"]
    stair_markers = ["^", "s", "D", "o"]
    for k, r in enumerate(results):
        lw = 2.4 if r["name"] == "full MSDiffAug" else 1.4
        ls = '--' if r["name"] == "Spectral overlap" else '-'
        color = ("#179d14" if r["name"] == "full MSDiffAug"
                 else stair_colors[k % 4])
        ax.plot(np.arange(len(BIN_LABELS)), r["bin_meds"],
                marker=stair_markers[k % 4], ms=4, lw=lw, ls=ls,
                color=color, label=r["name"])
        ax.errorbar(np.arange(len(BIN_LABELS)), r["bin_meds"],
                    yerr=np.array(bin_cis[r["name"]]).T, fmt="none",
                    ecolor=color, lw=0.8, capsize=2, zorder=5)
    ax.set_xticks(np.arange(len(BIN_LABELS)))
    ax.set_xticklabels(BIN_LABELS, fontsize=9)
    ax.tick_params(axis='y', labelsize=9)
    ax.set_xlabel('Structural similarity', fontsize=11)
    ax.set_ylabel('Median embedding cosine', fontsize=11)
    ax.legend(fontsize=10.5, framealpha=0.7)
    ax.grid(alpha=0.25)

    top_all = max((r["bin_meds"][k] + bin_cis[r["name"]][k][1]
                   for r in results for k in range(len(BIN_LABELS))
                   if not np.isnan(r["bin_meds"][k])), default=0.5)
    ax.set_ylim(bottom=0, top=top_all + 0.05)

    fig.tight_layout()
    for ext, kwargs in (("pdf", {}), ("png", {"dpi": 300})):
        out = os.path.join(_HERE, f"ablation_struct.{ext}")
        fig.savefig(out, **kwargs)
        print(f"saved: {out}")

    # ---- 结构感知 AUC 柱状图 (analog vs unrelated) ----
    fig2, ax2 = plt.subplots(figsize=(5.5, 4.6))
    names = [r["name"] for r in results]
    aucs = [r["auc"] for r in results]
    auc_colors = ["#F2B8AE", "#EDDDAB", "#C2E5CF", "#74A9A5"]
    yerrs = np.array([r["auc_ci"] for r in results]).T
    bars = ax2.bar(np.arange(len(names)), aucs, width=0.62,
                   color=[auc_colors[k % 4] for k in range(len(names))],
                   edgecolor="black", linewidth=0.6,
                   yerr=yerrs, capsize=3,
                   error_kw=dict(lw=0.8, ecolor="black"))
    for b, v in zip(bars, aucs):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=10)
    ax2.axhline(0.5, ls="--", lw=1, color="black", alpha=0.5)
    disp = {"Spectral overlap": "Spectral\noverlap",
            "Masking Aug": "Masking\nAug",
            "Random Perturbation Aug": "Random\nPerturbation Aug",
            "VAE Aug": "VAE\nAug",
            "full MSDiffAug": "full\nMSDiffAug"}
    ax2.set_xticks(np.arange(len(names)))
    ax2.set_xticklabels([disp.get(n, n) for n in names], fontsize=9)
    ax2.tick_params(axis='y', labelsize=9)
    ax2.set_ylabel("AUC", fontsize=11)
    ax2.set_ylim(0.5, 1.03)
    ax2.grid(alpha=0.25, axis="y")

    fig2.tight_layout()
    for ext, kwargs in (("pdf", {}), ("png", {"dpi": 300})):
        out = os.path.join(_HERE, f"ablation_auc.{ext}")
        fig2.savefig(out, **kwargs)
        print(f"saved: {out}")

    # ---- 逐对 CSV (补充材料) ----
    out_csv = os.path.join(_HERE, "emb_structure_ablation_pairs.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "pair_type", "idx1", "idx2",
                         "tanimoto", "emb_cos", "n_spec_pairs"])
        for r in results:
            cos_all = r["cos_all"]
            for (a, b), t, c in zip(pairs_arr, tan_all, cos_all):
                band = ("analog" if t >= _T_ANA
                        else "related" if t >= _T_REL else "unrelated")
                writer.writerow([r["name"], band, int(a), int(b),
                                 f"{t:.6f}", f"{c:.6f}",
                                 len(spec_of[int(a)]) * len(spec_of[int(b)])])
    print(f"saved: {out_csv}")


if __name__ == "__main__":
    main()
