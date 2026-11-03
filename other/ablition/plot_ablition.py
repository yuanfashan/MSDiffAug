# -*- coding: utf-8 -*-
"""recall-Tanimoto 曲线 (MSDiffAug vs MSBert 对比): 扫描嵌入相似度阈值,
观察覆盖率与结构相似度的权衡。

数据来源: 两个模型的 top-1 类似物检索结果 CSV (top_analogs05.csv,
由 test_tanimoto.py 生成, 含 emb_sim / tanimoto 列), 同一查询集。

- 横坐标 recall (%): 嵌入相似度 >= 阈值的查询占比 (有效部分 / 总查询数),
  从 100% 到 0 绘制 (阈值从低到高)
- 纵坐标: 有效部分的平均真实结构相似度 (Morgan/Tanimoto)
- 曲线越高: 嵌入相似度与真实结构相似度的一致性越好
"""
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 服务器无显示环境
import matplotlib.pyplot as plt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_PATHS = {
    "fullMSDiffAug": "/home/ausu/projects/yfs/MSDiffAugServer/ablition/fullMSDiffAug/fullMSDiffAug.csv",
    "NOCL": "/home/ausu/projects/yfs/MSDiffAugServer/ablition/NOCL/NO_CL.csv",
    "NOMP": "/home/ausu/projects/yfs/MSDiffAugServer/ablition/NOMP/NOMP.csv",
    "CAug": "/home/ausu/projects/yfs/MSDiffAugServer/ablition/CAug/MSBET.csv",
}
OUT_PATH = os.path.join(_SCRIPT_DIR, "ablition_rel_tamo.png")

N_THRESHOLDS = 200   # 每条曲线的阈值扫描点数

COLORS = {
    "fullMSDiffAug": "#4a7dba",
    "NOCL": "#ffd680",
    "CAug": "#fa874f",
    "NOMP": "#d93026",
}


def load_results(csv_path):
    # 谱库中混有非 SMILES 编号的条目 (如 "126041"), 其 tanimoto 为 nan,
    # 无法计算真实结构相似度, 整行跳过 (不进入 recall 分母与均值)
    emb_sims, tanimotos = [], []
    n_skipped = 0
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            emb, tani = float(row["emb_sim"]), float(row["tanimoto"])
            if np.isnan(emb) or np.isnan(tani):
                n_skipped += 1
                continue
            emb_sims.append(emb)
            tanimotos.append(tani)
    return emb_sims, tanimotos, n_skipped


def recall_curve(emb_sims, tanimotos, n_thresholds=N_THRESHOLDS):
    """阈值从低到高扫描, 返回 (recalls[%], mean_sims)。"""
    n = len(emb_sims)
    lo, hi = min(emb_sims), max(emb_sims)
    thresholds = [lo + (hi - lo) * i / (n_thresholds - 1) for i in range(n_thresholds)]
    recalls, mean_sims = [], []
    for t in thresholds:
        valid = [s for e, s in zip(emb_sims, tanimotos) if e >= t]
        if not valid:
            continue
        recalls.append(len(valid) / n * 100)
        mean_sims.append(sum(valid) / len(valid))
    return recalls, mean_sims


def main():
    plt.figure(figsize=(7, 5))
    for label, path in CSV_PATHS.items():
        if not os.path.exists(path):
            print(f"跳过 (文件不存在): {path}")
            continue
        emb_sims, tanimotos, n_skipped = load_results(path)
        recalls, mean_sims = recall_curve(emb_sims, tanimotos)
        highlight = label == "MSDiffAug"
        plt.plot(recalls, mean_sims, "-", color=COLORS[label],
                 linewidth=1.5,
                 zorder=10 if highlight else 5, label=label)
        print(f"{label}: {len(emb_sims)} 条查询 (跳过 {n_skipped} 条无效) | "
              f"recall=100% 平均 Tanimoto="
              f"{mean_sims[0]:.4f} | 最低 recall={recalls[-1]:.1f}% 平均 Tanimoto="
              f"{mean_sims[-1]:.4f}")

    plt.gca().invert_xaxis()  # 横坐标从 100% 到 0
    plt.xlabel("Recall (%)")
    plt.ylabel("ATS@1")
    plt.legend(fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=200)
    print(f"图已保存: {OUT_PATH}")


if __name__ == "__main__":
    main()
