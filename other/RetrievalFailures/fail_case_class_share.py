import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from case_feature_analysis import features
from class_model_metrics import is_long_chain, is_polar, DISPLAY

FAIL_CSV = "/home/ausu/projects/yfs/MSDiffAugServer/result_statistic/all_models_fail_cases.csv"


def main(path=None):
    path = path or (sys.argv[1] if len(sys.argv) > 1 else FAIL_CSV)
    df = pd.read_csv(path, encoding="utf-8-sig")
    print(f"fail cases: {len(df)}")

    df["_f"] = df["query_smiles"].astype(str).map(features)
    n_bad = df["_f"].isna().sum()
    if n_bad:
        print(f"unparseable query_smiles: {n_bad}, excluded")
    df = df.dropna(subset=["_f"])

    long_chain = pd.Series([is_long_chain(f) for f in df["_f"]], index=df.index)
    polar = pd.Series([is_polar(f) for f in df["_f"]], index=df.index)

    for name, mask in [("long_chain", long_chain), ("polar", polar)]:
        n = int(mask.sum())
        print(f"{DISPLAY[name]}: {n} / {len(df)} = {n / len(df) * 100:.4f}%")

    both = int((long_chain & polar).sum())
    neither = int((~(long_chain | polar)).sum())
    print(f"overlap (both classes): {both} / {len(df)} = {both / len(df) * 100:.4f}%")
    print(f"neither: {neither} / {len(df)} = {neither / len(df) * 100:.4f}%")


if __name__ == "__main__":
    main()
