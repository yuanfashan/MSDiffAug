# -*- coding: utf-8 -*-
"""从各模型 per-query 谱库检索结果出发, 重新生成两张失败案例表格的数据,
全部输出到 result_statistic/csvs/ (新目录, 不覆盖旧文件):

  tab:fail_class_share  结构类在 background / shared / msdiffaug fail / exclusive 中的占比
  tab:decoy_type        exclusive vs shared 的 top-1 decoy 类型 + Fisher 精确检验

输入: main() 中 model_csvs 显式给出的五个模型的 per-query CSV 绝对路径
  (与 msdiffaug_pr_libsearch_per_query.csv 同格式, 含 query_idx / query_smiles /
   top1_smiles / hit1 / ats1 等列), 先按 query_idx 合并再计算。
主模型取 model_csvs 的第一个键 (MSDiffAug); 换 run 时改 main() 里对应路径即可。
"""
import os
import sys

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import DataStructs, rdFingerprintGenerator, rdMolDescriptors
from scipy.stats import fisher_exact

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from case_feature_analysis import features

RDLogger.DisableLog("rdApp.error")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "result_statistic", "csvs")

POLAR_GROUPS = ["carboxyl", "phenol", "phosphate", "quatN"]
LONG_CHAIN_MIN = 8

FP = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# 论文表 1 的结构类, 顺序与 LaTeX 一致
CLASSES = [
    ("Acyclic", lambda f: f["nRings"] == 0),
    ("Chain >= 12", lambda f: f["longestChain"] >= 12),
    ("Phosphate", lambda f: f["phosphate"] > 0),
    ("Amino-acid-like", lambda f: f["aminoacid_like"] > 0),
    ("Long chain >= 8", lambda f: f["longestChain"] >= LONG_CHAIN_MIN),
    ("Polar", lambda f: f["LogP"] < 1.0 and any(f[g] > 0 for g in POLAR_GROUPS)),
    ("Carboxyl", lambda f: f["carboxyl"] > 0),
    ("LogP < 1", lambda f: f["LogP"] < 1.0),
    ("TPSA > 100", lambda f: f["TPSA"] > 100),
    ("Macrocycle", lambda f: f["macrocycle"] > 0),
    ("Phenol", lambda f: f["phenol"] > 0),
    ("Sugar moiety", lambda f: f["nSugar"] >= 1),
]

# 论文注释中提及但未列入正文的类别
EXTRA_CLASSES = [
    ("Halogen", lambda f: f["halogen"] > 0),
    ("Sulfonyl", lambda f: f["sulfonyl"] > 0),
    ("Quaternary ammonium", lambda f: f["quatN"] > 0),
    ("Salt", lambda f: f["salt_ion"] > 0),
    ("Amide", lambda f: f["amide"] > 0),
    ("Fused rings", lambda f: f["fusedAtoms"] > 0),
    ("Aromatic rings >= 2", lambda f: f["nAromRings"] >= 2),
    ("Aliphatic rings >= 1", lambda f: f["nAliphRings"] >= 1),
    ("Ring atom frac > 0.5", lambda f: f["ringAtomFrac"] > 0.5),
    ("MW < 200", lambda f: f["MW"] < 200),
    ("MW > 400", lambda f: f["MW"] > 400),
    ("Chiral >= 1", lambda f: f["Chiral"] >= 1),
]

# 表 2 的 decoy 类型: 类别 -> [(decoy 类型名, 判据函数(row_flags, qf) -> bool)]
DECOY_TYPES = {
    "Phenol": [("Same-formula isomer", lambda fl, qf: fl["same_formula"])],
    "Macrocycle": [
        ("Same-formula isomer", lambda fl, qf: fl["same_formula"]),
        ("Top-1 is macrocycle", lambda fl, qf: fl["top1_macrocycle"] > 0),
    ],
    "Sugar moiety": [
        ("Same sugar, different formula",
         lambda fl, qf: (fl["top1_nSugar"] > 0)
         and (fl["top1_nSugar"] == qf["nSugar"]) and not fl["same_formula"]),
        ("Longest acyclic carbon chain >= 8",
         lambda fl, qf: qf["longestChain"] >= LONG_CHAIN_MIN),
    ],
}


def add_features(df, name):
    df["_f"] = df["query_smiles"].astype(str).map(features)
    n_bad = int(df["_f"].isna().sum())
    if n_bad:
        print(f"[{name}] unparseable query_smiles: {n_bad}, excluded")
    return df.dropna(subset=["_f"]).reset_index(drop=True)


def _mol(smi):
    return Chem.MolFromSmiles(str(smi)) if smi else None


def _fp(smi):
    mol = _mol(smi)
    return FP.GetFingerprint(mol) if mol is not None else None


def _formula(smi):
    mol = _mol(smi)
    return rdMolDescriptors.CalcMolFormula(mol) if mol is not None else None


def decoy_flags(q_smiles, t_smiles):
    tf = features(t_smiles) if t_smiles else None
    out = {"top1_parseable": tf is not None, "tanimoto": float("nan"),
           "same_formula": False, "top1_nSugar": -1, "top1_phenol": -1,
           "top1_macrocycle": -1}
    if tf is None:
        return out
    fq, ft = _fp(q_smiles), _fp(t_smiles)
    if fq is not None and ft is not None:
        out["tanimoto"] = DataStructs.TanimotoSimilarity(fq, ft)
    out["same_formula"] = bool(_formula(q_smiles) == _formula(t_smiles))
    out["top1_nSugar"] = tf["nSugar"]
    out["top1_phenol"] = tf["phenol"]
    out["top1_macrocycle"] = tf["macrocycle"]
    return out


def class_shares_table(dfs):
    rows = []
    for cname, fn in CLASSES + EXTRA_CLASSES:
        row = {"class": cname}
        for sname, df in dfs.items():
            mask = pd.Series([fn(f) for f in df["_f"]], index=df.index)
            row[f"{sname}_n"] = int(mask.sum())
            row[f"{sname}_pct"] = float(mask.mean() * 100)
        rows.append(row)
    return pd.DataFrame(rows)


def set_summary_rows(dfs):
    rows = []
    for sname, df in dfs.items():
        lc = pd.Series([f["longestChain"] >= LONG_CHAIN_MIN for f in df["_f"]],
                       index=df.index)
        pol = pd.Series(
            [f["LogP"] < 1.0 and any(f[g] > 0 for g in POLAR_GROUPS)
             for f in df["_f"]], index=df.index)
        n = len(df)
        both = int((lc & pol).sum())
        union = int((lc | pol).sum())
        rows.append({
            "set": sname, "n": n,
            "long_chain_n": int(lc.sum()), "long_chain_pct": lc.mean() * 100,
            "polar_n": int(pol.sum()), "polar_pct": pol.mean() * 100,
            "overlap_n": both, "overlap_pct": both / n * 100,
            "union_n": union, "union_pct": union / n * 100,
            "neither_n": n - union, "neither_pct": (n - union) / n * 100,
        })
    return pd.DataFrame(rows)


def decoy_table(excl, shared):
    """excl/shared 需带 _f; 逐类别算 decoy 类型 + Fisher 检验"""
    rows = []
    per_rows = {}
    class_pred = {c: fn for c, fn in CLASSES + EXTRA_CLASSES}
    for cname, dtype_fns in DECOY_TYPES.items():
        pred = class_pred[cname]
        sub_excl = excl[pd.Series([pred(f) for f in excl["_f"]], index=excl.index)]
        sub_shared = shared[pd.Series([pred(f) for f in shared["_f"]],
                                      index=shared.index)]
        sub_excl = sub_excl.assign(group="exclusive")
        sub_shared = sub_shared.assign(group="shared")
        sub = pd.concat([sub_excl, sub_shared], ignore_index=True)
        flags = pd.DataFrame(
            [decoy_flags(r["query_smiles"], r["top1_smiles"])
             for _, r in sub.iterrows()], index=sub.index)
        parsed = flags["top1_parseable"]
        se, ss = flags[parsed & (sub["group"] == "exclusive")], \
                 flags[parsed & (sub["group"] == "shared")]
        n_excl, n_shared = len(se), len(ss)
        for dname, fn in dtype_fns:
            qf_excl = sub.loc[parsed & (sub["group"] == "exclusive"), "_f"]
            qf_shared = sub.loc[parsed & (sub["group"] == "shared"), "_f"]
            a = int(pd.Series([fn(fl, qf) for fl, qf in
                               zip(se.to_dict("records"),
                                   qf_excl.tolist())]).sum())
            c = int(pd.Series([fn(fl, qf) for fl, qf in
                               zip(ss.to_dict("records"),
                                   qf_shared.tolist())]).sum())
            b, d = n_excl - a, n_shared - c
            odds, p_two = fisher_exact([[a, b], [c, d]],
                                       alternative="two-sided")
            _, p_greater = fisher_exact([[a, b], [c, d]],
                                        alternative="greater")
            rows.append({
                "class": cname, "decoy_type": dname,
                "excl_n": a, "excl_denom": n_excl,
                "excl_pct": a / n_excl * 100 if n_excl else float("nan"),
                "shared_n": c, "shared_denom": n_shared,
                "shared_pct": c / n_shared * 100 if n_shared else float("nan"),
                "fisher_two_sided_p": p_two,
                "fisher_greater_p": p_greater,
                "odds_ratio": odds,
            })
        keep = pd.concat([sub, flags], axis=1)
        per_rows[cname] = keep.drop(columns=["_f"])
    return pd.DataFrame(rows), per_rows


def main():
    model_csvs = {
        "MSDiffAug": "/home/ausu/projects/yfs/MSDiffAugServer/ablition/NOCL/msdiffaug_libsearch_per_query.csv",
        # "MSDiffAug8": "/home/ausu/projects/yfs/MSDiffAugServer/ablition5/msdiffaug/MSDiffAug_gaq2_1.csv"
        "MSBert": "/home/ausu/projects/yfs/MSDiffAugServer/ablition5/benchmark_out/msbert/msbert_libsearch_per_query.csv",
        "MS2DeepScore": "/home/ausu/projects/yfs/MSDiffAugServer/ablition5/benchmark_out/ms2deepscore/ms2deepscore_libsearch_per_query.csv",
        "Spec2Vec": "/home/ausu/projects/yfs/MSDiffAugServer/ablition5/benchmark_out/spec2vec/spec2vec_libsearch_per_query.csv",
        "Binned MS/MS": "/home/ausu/projects/yfs/MSDiffAugServer/ablition5/benchmark_out/msms/msms_libsearch_per_query.csv",
    }
    main_model = next(iter(model_csvs))
    # 合并列名用项目统一的小写模型前缀, 与历史 CSV (msbert_hit1 等) 一致
    col_prefix = {
        "MSDiffAug": "msdiffaug",
        "MSBert": "msbert",
        "MS2DeepScore": "ms2deepscore",
        "Spec2Vec": "spec2vec",
        "Binned MS/MS": "msms",
    }
    key = "query_idx"
    keep_cols = [key, "query_smiles", "query_ik", "top1_smiles", "top1_ik",
                 "top1_emb_sim", "hit1", "hit5", "hit10", "ats1", "ats5",
                 "ats10"]

    os.makedirs(OUT_DIR, exist_ok=True)

    frames = {}
    for m, p in model_csvs.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing per-query csv for {m}: {p}")
        df = pd.read_csv(p, encoding="utf-8-sig")
        assert not df[key].duplicated().any(), f"{m}: duplicated query_idx"
        frames[m] = df
        print(f"{m}: {len(df)} queries  ({p})")

    other_models = [m for m in model_csvs if m != main_model]
    merged = frames[main_model][keep_cols].copy()
    for m in other_models:
        pfx = col_prefix[m]
        sub = frames[m][[key, "hit1", "ats1"]].rename(
            columns={"hit1": f"{pfx}_hit1", "ats1": f"{pfx}_ats1"})
        merged = merged.merge(sub, on=key, how="left")
    other_cols = [f"{col_prefix[m]}_hit1" for m in other_models] + \
                 [f"{col_prefix[m]}_ats1" for m in other_models]
    n_missing = merged[other_cols].isna().any(axis=1).sum()
    if n_missing:
        print(f"warning: {n_missing} queries missing in some other model, "
              f"excluded")
        merged = merged.dropna(subset=other_cols)

    merged = add_features(merged, "merged")
    other_hit = [f"{col_prefix[m]}_hit1" for m in other_models]
    msdiff_fail = merged[merged["hit1"] == 0].reset_index(drop=True)
    af = merged[(merged[["hit1"] + other_hit] == 0).all(axis=1)] \
        .reset_index(drop=True)
    excl = msdiff_fail[msdiff_fail[other_hit].max(axis=1) == 1] \
        .reset_index(drop=True)

    print(f"\n=== set sizes ===")
    print(f"background (all queries): {len(merged)}")
    print(f"shared failures (all five hit1=0): {len(af)}")
    print(f"msdiffaug failures (hit1=0): {len(msdiff_fail)}")
    print(f"exclusive failures: {len(excl)}")
    assert len(excl) == len(msdiff_fail) - len(af), \
        "exclusive != msdiffaug failures minus shared failures"
    print("sanity checks passed")

    merged.drop(columns=["_f"]).to_csv(
        os.path.join(OUT_DIR, "merged_per_query.csv"), index=False,
        encoding="utf-8-sig")

    dfs = {"background": merged, "shared": af, "msdiffaug_fail": msdiff_fail,
           "exclusive": excl}

    # ---- 表 1: 结构类占比 ----
    tbl = class_shares_table(dfs)
    paper_names = [c for c, _ in CLASSES]
    tbl_paper = tbl[tbl["class"].isin(paper_names)].copy()
    order = {c: i for i, c in enumerate(paper_names)}
    tbl_paper = tbl_paper.sort_values(
        "class", key=lambda s: s.map(order)).reset_index(drop=True)
    n_row = pd.DataFrame([{"class": "N", **{
        f"{s}_n": len(df) for s, df in dfs.items()}}])
    tbl_paper_out = pd.concat([tbl_paper, n_row], ignore_index=True)
    tbl_paper_out.to_csv(os.path.join(OUT_DIR, "fail_class_share.csv"),
                         index=False)
    tbl.to_csv(os.path.join(OUT_DIR, "class_shares_full.csv"), index=False)

    print("\n=== tab:fail_class_share (%) ===")
    cols = ["class", "background_pct", "shared_pct", "msdiffaug_fail_pct",
            "exclusive_pct"]
    print(tbl_paper_out[cols].round(2).to_string(index=False))
    print(f"  (counts row: {len(merged)} / {len(af)} / {len(msdiff_fail)} / "
          f"{len(excl)})")

    # ---- 长链/极性并集与重叠 (表 1 注释用) ----
    ssum = set_summary_rows(dfs)
    ssum.to_csv(os.path.join(OUT_DIR, "set_summary.csv"), index=False)
    print("\n=== set summary (long chain / polar / union) ===")
    print(ssum.round(2).to_string(index=False))

    # ---- 表 2: decoy 类型 + Fisher ----
    dtab, per_rows = decoy_table(excl, af)
    dtab.to_csv(os.path.join(OUT_DIR, "decoy_type.csv"), index=False)
    print("\n=== tab:decoy_type ===")
    for _, r in dtab.iterrows():
        print(f"  {r['class']:<15} {r['decoy_type']:<32} "
              f"excl {r['excl_n']}/{r['excl_denom']} "
              f"({r['excl_pct']:.1f}%)  shared {r['shared_n']}/"
              f"{r['shared_denom']} ({r['shared_pct']:.1f}%)  "
              f"p_two={r['fisher_two_sided_p']:.4f}  "
              f"p_greater={r['fisher_greater_p']:.4f}  "
              f"OR={r['odds_ratio']:.2f}")

    for cname, df in per_rows.items():
        fname = cname.lower().replace(" ", "_").replace(">=", "ge")
        df.to_csv(os.path.join(OUT_DIR, f"decoy_rows_{fname}.csv"),
                  index=False, encoding="utf-8-sig")

    # ---- 失败集合成员标签 ----
    af_lab = af.assign(group="shared")
    excl_lab = excl.assign(group="exclusive")
    pd.concat([af_lab, excl_lab], ignore_index=True).drop(
        columns=["_f"]).to_csv(os.path.join(OUT_DIR, "fail_sets.csv"),
                               index=False, encoding="utf-8-sig")

    print(f"\nsaved all outputs to: {OUT_DIR}")
    for f in sorted(os.listdir(OUT_DIR)):
        print(f"  {f}")


if __name__ == "__main__":
    main()
