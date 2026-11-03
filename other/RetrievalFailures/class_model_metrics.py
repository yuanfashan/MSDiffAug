import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from library_search import MODEL_CSVS, KEY, MAIN_MODEL, OTHER_MODELS, OUT_DIR
from case_feature_analysis import features

METRICS = ["hit1", "hit5", "hit10", "ats1", "ats5", "ats10"]
ALL_MODELS = [MAIN_MODEL] + OTHER_MODELS

# 类别定义（可调）
LONG_CHAIN_MIN = 8      # 长链脂类: 最长非环碳链 >= 8（改 12 即链>=12 子类）
POLAR_MAX_LOGP = 1.0    # 极性分子: LogP < 1 且含下列极性官能团之一
POLAR_GROUPS = ["carboxyl", "phenol", "phosphate", "quatN"]

DISPLAY = {
    "aminoacid_like": "氨基酸样 (amino-acid-like motif)",
    "aa_lipo": "脂酰氨基酸 (aa & long chain)",
    "aa_polar": "极性氨基酸/肽 (aa & polar, 非长链)",
    "aa_other": "其他氨基酸样 (aa, 非长链非极性)",
    "long_non_aa": "非氨基酸长链 (long chain, 非 aa)",
    "polar_non_aa": "非氨基酸极性 (polar, 非 aa 非长链)",
    "long_chain": f"长链脂类 (longestChain >= {LONG_CHAIN_MIN})",
    "polar": f"极性分子 (LogP < {POLAR_MAX_LOGP} 且含 {'/'.join(POLAR_GROUPS)})",
}


def is_long_chain(f):
    return f["longestChain"] >= LONG_CHAIN_MIN


def is_polar(f):
    return f["LogP"] < POLAR_MAX_LOGP and any(f[g] > 0 for g in POLAR_GROUPS)


def is_aa(f):
    return f["aminoacid_like"] > 0


def load_merged():
    frames = {m: pd.read_csv(p, encoding="utf-8-sig") for m, p in MODEL_CSVS.items()}
    base = frames[MAIN_MODEL][[KEY, "query_smiles"]].copy()
    for m in ALL_MODELS:
        sub = frames[m][[KEY] + METRICS].rename(
            columns={c: f"{m}_{c}" for c in METRICS})
        base = base.merge(sub, on=KEY, how="left")
    metric_cols = [f"{m}_{c}" for m in ALL_MODELS for c in METRICS]
    n_missing = base[metric_cols].isna().any(axis=1).sum()
    if n_missing:
        print(f"warning: {n_missing} queries missing in some model, excluded")
        base = base.dropna(subset=metric_cols)
    return base


def class_metrics(base, mask):
    sub = base[mask]
    block = pd.DataFrame(
        {m: {c: sub[f"{m}_{c}"].mean() for c in METRICS} for m in ALL_MODELS}
    ).T
    block[["hit1", "hit5", "hit10"]] *= 100
    return sub, block


def main():
    base = load_merged()
    base["_f"] = base["query_smiles"].astype(str).map(features)
    n_bad = base["_f"].isna().sum()
    if n_bad:
        print(f"unparseable query_smiles: {n_bad}, excluded")
    base = base.dropna(subset=["_f"])

    lc = pd.Series([is_long_chain(f) for f in base["_f"]], index=base.index)
    pol = pd.Series([is_polar(f) for f in base["_f"]], index=base.index)
    aa = pd.Series([is_aa(f) for f in base["_f"]], index=base.index)

    masks = {
        "aminoacid_like": aa,
        "aa_lipo": aa & lc,
        "aa_polar": aa & pol & ~lc,
        "aa_other": aa & ~lc & ~pol,
        "long_non_aa": lc & ~aa,
        "polar_non_aa": pol & ~aa & ~lc,
        "long_chain": lc,
        "polar": pol,
    }

    ref, ref_block = class_metrics(base, pd.Series(True, index=base.index))
    print(f"=== ALL (reference): n={len(ref)} ===")
    print(ref_block.round(4).to_string())

    print("\n=== 各类样本数 (全体查询) ===")
    for cname, mask in masks.items():
        print(f"  {DISPLAY[cname]:<46} n={int(mask.sum())} ({mask.mean() * 100:.2f}%)")

    overlap = int((lc & pol).sum())
    print(f"\noverlap between long_chain and polar: {overlap}")
    check = int((masks["aa_lipo"] | masks["aa_polar"] | masks["aa_other"]).sum())
    print(f"check: aa_lipo ∪ aa_polar ∪ aa_other = {check} (aa = {int(aa.sum())})")

    for cname, mask in masks.items():
        sub, block = class_metrics(base, mask)
        print(f"\n=== {DISPLAY[cname]}: n={len(sub)} ===")
        print(block.round(4).to_string())
        block.round(4).to_csv(
            os.path.join(OUT_DIR, f"class_{cname}_model_metrics.csv"))
        sub[[KEY, "query_smiles"]].to_csv(
            os.path.join(OUT_DIR, f"class_{cname}_queries.csv"), index=False)
        print(f"saved: class_{cname}_model_metrics.csv, class_{cname}_queries.csv")


if __name__ == "__main__":
    main()
