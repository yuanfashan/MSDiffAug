import os
import sys
from collections import Counter

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from library_search import MODEL_CSVS, KEY, MAIN_MODEL, OTHER_MODELS, KEEP_COLS, OUT_DIR

SMARTS = {
    "carboxyl": "[CX3](=O)[OX2H1,OX1H0-]",
    "phenol": "[OX2H1][c]",
    "amide": "[NX3][CX3](=[OX1])",
    "aminoacid_like": "[NX3;H1,H2][CX4][CX3](=O)[OX2H1,OX1H0-]",
    "quatN": "[NX4+]",
    "halogen": "[F,Cl,Br,I]",
    "salt_ion": "[Na+,K+,Cl-]",
    "phosphate": "[PX4]",
    "sulfonyl": "[SX4](=O)(=O)",
}
PATTERNS = {k: Chem.MolFromSmarts(v) for k, v in SMARTS.items()}

NUM_FEATURES = [
    "nRings", "nAromRings", "nAliphRings", "ringAtomFrac", "longestChain",
    "fusedAtoms", "nSugar", "MW", "LogP", "TPSA", "HBD", "HBA", "RotB",
    "HeavyAtoms", "nO", "nN", "Chiral",
]

BIN_FEATURES = {
    "has_carboxyl": lambda f: f["carboxyl"] > 0,
    "has_phenol": lambda f: f["phenol"] > 0,
    "has_amide": lambda f: f["amide"] > 0,
    "aminoacid_like": lambda f: f["aminoacid_like"] > 0,
    "has_sugar": lambda f: f["nSugar"] >= 1,
    "has_salt": lambda f: f["salt_ion"] > 0,
    "has_halogen": lambda f: f["halogen"] > 0,
    "has_quatN": lambda f: f["quatN"] > 0,
    "has_phosphate": lambda f: f["phosphate"] > 0,
    "has_sulfonyl": lambda f: f["sulfonyl"] > 0,
    "no_ring": lambda f: f["nRings"] == 0,
    "chain_ge8": lambda f: f["longestChain"] >= 8,
    "chain_ge12": lambda f: f["longestChain"] >= 12,
    "macrocycle": lambda f: f["macrocycle"] > 0,
    "fused_rings": lambda f: f["fusedAtoms"] > 0,
    "arom_rings_ge2": lambda f: f["nAromRings"] >= 2,
    "aliph_rings_ge1": lambda f: f["nAliphRings"] >= 1,
    "aliph_rings_ge2": lambda f: f["nAliphRings"] >= 2,
    "ring_frac_gt05": lambda f: f["ringAtomFrac"] > 0.5,
    "MW_lt200": lambda f: f["MW"] < 200,
    "MW_gt400": lambda f: f["MW"] > 400,
    "LogP_lt1": lambda f: f["LogP"] < 1,
    "TPSA_gt100": lambda f: f["TPSA"] > 100,
    "chiral_ge1": lambda f: f["Chiral"] >= 1,
}


def sugar_rings(mol):
    n = 0
    for ring in mol.GetRingInfo().AtomRings():
        if len(ring) not in (5, 6):
            continue
        if sum(1 for a in ring if mol.GetAtomWithIdx(a).GetAtomicNum() == 8) != 1:
            continue
        sub = 0
        for a in ring:
            for nb in mol.GetAtomWithIdx(a).GetNeighbors():
                if nb.GetIdx() not in ring and nb.GetAtomicNum() == 8:
                    sub += 1
        if sub >= 2:
            n += 1
    return n


def n_chiral(mol):
    return sum(1 for a in mol.GetAtoms()
               if a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED)


def n_arom_rings(mol):
    return sum(1 for r in mol.GetRingInfo().AtomRings()
               if all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in r))


def n_aliph_rings(mol):
    return sum(1 for r in mol.GetRingInfo().AtomRings()
               if not any(mol.GetAtomWithIdx(a).GetIsAromatic() for a in r))


def fused_atoms(mol):
    cnt = Counter()
    for r in mol.GetRingInfo().AtomRings():
        cnt.update(r)
    return sum(1 for v in cnt.values() if v >= 2)


def ring_atom_frac(mol):
    ring_atoms = {a for r in mol.GetRingInfo().AtomRings() for a in r}
    n = mol.GetNumHeavyAtoms()
    return len(ring_atoms) / n if n else 0.0


def macrocycle_count(mol):
    return sum(1 for r in mol.GetRingInfo().AtomRings() if len(r) >= 9)


def longest_carbon_chain(mol):
    """最长非环碳链长度（只数不在任何环上的碳原子）"""
    ring_atoms = {a for r in mol.GetRingInfo().AtomRings() for a in r}
    cs = [a.GetIdx() for a in mol.GetAtoms()
          if a.GetAtomicNum() == 6 and a.GetIdx() not in ring_atoms]
    if not cs:
        return 0
    cset = set(cs)
    adj = {i: [n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors()
               if n.GetIdx() in cset] for i in cs}
    best = 0
    seen = set()
    for s in cs:
        if s in seen:
            continue

        def bfs(start):
            dist = {start: 0}
            far = start
            q = [start]
            for u in q:
                for v in adj[u]:
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        q.append(v)
                        if dist[v] > dist[far]:
                            far = v
            return far, dist

        far1, _ = bfs(s)
        far2, dist = bfs(far1)
        seen.update(dist)
        best = max(best, dist[far2] + 1)
    return best


def features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    feats = {
        "MW": Descriptors.MolWt(mol),
        "LogP": Crippen.MolLogP(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "HBD": Lipinski.NumHDonors(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "RotB": Descriptors.NumRotatableBonds(mol),
        "HeavyAtoms": mol.GetNumHeavyAtoms(),
        "nO": sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 8),
        "nN": sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 7),
        "nRings": Descriptors.RingCount(mol),
        "nAromRings": n_arom_rings(mol),
        "nAliphRings": n_aliph_rings(mol),
        "ringAtomFrac": ring_atom_frac(mol),
        "longestChain": longest_carbon_chain(mol),
        "fusedAtoms": fused_atoms(mol),
        "macrocycle": macrocycle_count(mol),
        "nSugar": sugar_rings(mol),
        "Chiral": n_chiral(mol),
    }
    for name, pat in PATTERNS.items():
        feats[name] = len(mol.GetSubstructMatches(pat))
    return feats


def load_merged():
    frames = {m: pd.read_csv(p, encoding="utf-8-sig") for m, p in MODEL_CSVS.items()}
    merged = frames[MAIN_MODEL][KEEP_COLS].copy()
    for m in OTHER_MODELS:
        sub = frames[m][[KEY, "hit1", "ats1"]].rename(
            columns={"hit1": f"{m}_hit1", "ats1": f"{m}_ats1"})
        merged = merged.merge(sub, on=KEY, how="left")
    other_cols = [f"{m}_hit1" for m in OTHER_MODELS] + [f"{m}_ats1" for m in OTHER_MODELS]
    n_missing = merged[other_cols].isna().any(axis=1).sum()
    if n_missing:
        print(f"warning: {n_missing} queries missing in some other model, excluded")
        merged = merged.dropna(subset=other_cols)
    return merged


def analyze(merged, out_dir=None):
    out_dir = out_dir or OUT_DIR
    other_hit = [f"{m}_hit1" for m in OTHER_MODELS]

    df = merged.copy()
    df["_f"] = df["query_smiles"].astype(str).map(features)
    n_bad = df["_f"].isna().sum()
    if n_bad:
        print(f"unparseable query_smiles: {n_bad}, excluded")
    df = df.dropna(subset=["_f"])

    groups = {
        "success": df["hit1"].eq(1),
        "fail": df["hit1"].eq(0),
        "all_fail": (df[["hit1"] + other_hit].eq(0)).all(axis=1),
    }
    sizes = {k: int(v.sum()) for k, v in groups.items()}
    print(f"total parsed queries: {len(df)}")
    print(f"group sizes: {sizes}")
    print(f"msdiffaug hit1 success rate: {sizes['success'] / len(df) * 100:.1f}%")

    fdf = pd.DataFrame(df["_f"].tolist(), index=df.index)

    mean_rows = []
    for c in NUM_FEATURES:
        row = {"feature": c, "bg": fdf[c].mean()}
        for gname, g in groups.items():
            row[gname] = fdf.loc[g, c].mean()
        mean_rows.append(row)
    mean_df = pd.DataFrame(mean_rows)

    prev_rows = []
    for name, fn in BIN_FEATURES.items():
        col = pd.Series([fn(f) for f in df["_f"]], index=df.index)
        row = {"feature": name, "bg": col.mean() * 100}
        for gname, g in groups.items():
            row[gname] = col[g].mean() * 100
        prev_rows.append(row)
    prev_df = pd.DataFrame(prev_rows)

    print("\n=== 数值特征均值: bg | success | fail | all_fail ===")
    print(mean_df.round(2).to_string(index=False))
    print("\n=== 二值特征占比(%): bg | success | fail | all_fail ===")
    print(prev_df.round(1).to_string(index=False))

    # ---- 按结构类: 各模型命中率对比（绝对优势/相对优势/相对弱势/共同局限） ----
    model_cols = {MAIN_MODEL: "hit1"}
    model_cols.update({m: f"{m}_hit1" for m in OTHER_MODELS})
    all_fail_mask = groups["all_fail"]

    class_defs = dict(BIN_FEATURES)
    class_defs.update({
        "rings_0": lambda f: f["nRings"] == 0,
        "rings_1": lambda f: f["nRings"] == 1,
        "rings_2": lambda f: f["nRings"] == 2,
        "rings_3plus": lambda f: f["nRings"] >= 3,
        "arom_0": lambda f: f["nAromRings"] == 0,
        "arom_1": lambda f: f["nAromRings"] == 1,
        "arom_2plus": lambda f: f["nAromRings"] >= 2,
        "chain_0to1": lambda f: f["longestChain"] <= 1,
        "chain_2to3": lambda f: 2 <= f["longestChain"] <= 3,
        "chain_4to7": lambda f: 4 <= f["longestChain"] <= 7,
        "chain_8to11": lambda f: 8 <= f["longestChain"] <= 11,
        "chain_12plus": lambda f: f["longestChain"] >= 12,
        "MW_200to300": lambda f: 200 <= f["MW"] < 300,
        "MW_300to400": lambda f: 300 <= f["MW"] < 400,
        "MW_400to500": lambda f: 400 <= f["MW"] < 500,
        "MW_ge500": lambda f: f["MW"] >= 500,
        "LogP_lt0": lambda f: f["LogP"] < 0,
        "LogP_0to3": lambda f: 0 <= f["LogP"] < 3,
        "LogP_3to5": lambda f: 3 <= f["LogP"] < 5,
        "LogP_ge5": lambda f: f["LogP"] >= 5,
    })

    rows = []
    for name, fn in class_defs.items():
        mask = pd.Series([fn(f) for f in df["_f"]], index=df.index)
        if not mask.any():
            continue
        row = {"class": name, "n": int(mask.sum())}
        rates = {mname: df.loc[mask, col].mean() * 100 for mname, col in model_cols.items()}
        for mname, r in rates.items():
            row[mname] = round(r, 1)
        best_other = max(rates[m] for m in OTHER_MODELS)
        row["best_other"] = round(best_other, 1)
        row["delta_vs_best"] = round(rates[MAIN_MODEL] - best_other, 1)
        row["all_fail_rate"] = round(all_fail_mask[mask].mean() * 100, 1)
        rows.append(row)

    ov_rates = {mname: df[col].mean() * 100 for mname, col in model_cols.items()}
    overall = {"class": "ALL", "n": len(df)}
    for mname, r in ov_rates.items():
        overall[mname] = round(r, 1)
    overall["best_other"] = round(max(ov_rates[m] for m in OTHER_MODELS), 1)
    overall["delta_vs_best"] = round(
        ov_rates[MAIN_MODEL] - max(ov_rates[m] for m in OTHER_MODELS), 1)
    overall["all_fail_rate"] = round(all_fail_mask.mean() * 100, 1)
    rows.insert(0, overall)

    rate_df = pd.DataFrame(rows)

    print("\n=== 各模型整体命中率 ===")
    print("  " + "  ".join(f"{m}: {ov_rates[m]:.1f}%" for m in model_cols))

    def show(title, sub):
        print(f"\n=== {title} ===")
        if sub.empty:
            print("  (无)")
            return
        for _, r in sub.iterrows():
            print(f"  {r['class']:<18} n={r['n']:>6}  msdiffaug={r['msdiffaug']:5.1f}%  "
                  f"best={r['best_other']:5.1f}%  delta={r['delta_vs_best']:+5.1f}  "
                  f"all_fail={r['all_fail_rate']:5.1f}%")

    min_n = max(1, len(df) // 1000)
    tv = rate_df[(rate_df["class"] != "ALL") & (rate_df["n"] >= min_n)]
    show("绝对优势: msdiffaug 命中率最高的结构类 (top10)",
         tv.sort_values("msdiffaug", ascending=False).head(10))
    show("相对优势: msdiffaug 领先最强基线最多的结构类 (top10)",
         tv[tv["delta_vs_best"] > 0].sort_values("delta_vs_best", ascending=False).head(10))
    show("相对弱势: msdiffaug 落后最强基线最多的结构类 (top10)",
         tv[tv["delta_vs_best"] < 0].sort_values("delta_vs_best", ascending=True).head(10))
    show("共同局限: 所有模型均失败比例最高的结构类 (top10)",
         tv.sort_values("all_fail_rate", ascending=False).head(10))

    mean_df.to_csv(os.path.join(out_dir, "case_feature_means.csv"), index=False)
    prev_df.to_csv(os.path.join(out_dir, "case_feature_prevalence.csv"), index=False)
    rate_df.to_csv(os.path.join(out_dir, "case_class_model_rates.csv"), index=False)
    print(f"\nsaved: {os.path.join(out_dir, 'case_feature_means.csv')}")
    print(f"saved: {os.path.join(out_dir, 'case_feature_prevalence.csv')}")
    print(f"saved: {os.path.join(out_dir, 'case_class_model_rates.csv')}")
    return mean_df, prev_df, rate_df


if __name__ == "__main__":
    analyze(load_merged())
