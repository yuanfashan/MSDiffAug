import os

import pandas as pd

BASE = "/home/ausu/projects/yfs/MSDiffAugServer/ablition5/benchmark_out"
BASE1 = "/home/ausu/projects/yfs/MSDiffAugServer/ablition"

MODEL_CSVS = {
    # "msdiffaug": os.path.join(BASE, "msdiffaug_pr", "msdiffaug_pr_libsearch_per_query.csv"),
    "msdiffaug": os.path.join(BASE1, "NOCL", "msdiffaug_libsearch_per_query.csv"),
    "msbert": os.path.join(BASE, "msbert", "msbert_libsearch_per_query.csv"),
    "ms2deepscore": os.path.join(BASE, "ms2deepscore", "ms2deepscore_libsearch_per_query.csv"),
    "spec2vec": os.path.join(BASE, "spec2vec", "spec2vec_libsearch_per_query.csv"),
    "msms": os.path.join(BASE, "msms", "msms_libsearch_per_query.csv"),
}

KEY = "query_idx"
MAIN_MODEL = "msdiffaug"
OTHER_MODELS = [m for m in MODEL_CSVS if m != MAIN_MODEL]

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

KEEP_COLS = [KEY, "query_smiles", "query_ik", "top1_smiles", "top1_ik",
             "top1_emb_sim", "hit1", "hit5", "hit10", "ats1", "ats5", "ats10"]


def main():
    frames = {m: pd.read_csv(p, encoding="utf-8-sig") for m, p in MODEL_CSVS.items()}
    for m, df in frames.items():
        print(f"{m}: {len(df)} queries")

    merged = frames[MAIN_MODEL][KEEP_COLS].copy()
    for m in OTHER_MODELS:
        sub = frames[m][[KEY, "hit1", "ats1"]].rename(
            columns={"hit1": f"{m}_hit1", "ats1": f"{m}_ats1"})
        merged = merged.merge(sub, on=KEY, how="left")

    other_hit = [f"{m}_hit1" for m in OTHER_MODELS]
    other_ats = [f"{m}_ats1" for m in OTHER_MODELS]

    n_missing = merged[other_hit + other_ats].isna().any(axis=1).sum()
    if n_missing:
        print(f"warning: {n_missing} queries missing in some other model, excluded")
        merged = merged.dropna(subset=other_hit + other_ats)

    # hit1 为 0/1，严格优于所有模型等价于 msdiffaug=1 且其余全为 0
    win_hit1 = merged["hit1"] > merged[other_hit].max(axis=1)
    win_ats1 = merged["ats1"] > merged[other_ats].max(axis=1)
    all_fail = (merged[["hit1"] + other_hit] == 0).all(axis=1)
    # 所有其他模型 hit1=1 而 msdiffaug hit1=0
    msdiffaug_fail_others_win = (merged["hit1"] == 0) & (merged[other_hit] == 1).all(axis=1)
    # msdiffaug hit1=0（全部失败案例）
    msdiffaug_fail = merged["hit1"] == 0

    merged[win_hit1].to_csv(os.path.join(OUT_DIR, "msdiffaug_win_hit1_cases.csv"), index=False)
    merged[win_ats1].to_csv(os.path.join(OUT_DIR, "msdiffaug_win_ats1_cases.csv"), index=False)
    merged[all_fail].to_csv(os.path.join(OUT_DIR, "all_models_fail_cases.csv"), index=False)
    merged[msdiffaug_fail_others_win].to_csv(
        os.path.join(OUT_DIR, "msdiffaug_fail_others_win_cases.csv"), index=False)
    merged[msdiffaug_fail].to_csv(
        os.path.join(OUT_DIR, "msdiffaug_fail_cases.csv"), index=False)

    print(f"total: {len(merged)}")
    print(f"msdiffaug hit1 优于所有模型: {win_hit1.sum()} -> msdiffaug_win_hit1_cases.csv")
    print(f"msdiffaug ats1 优于所有模型: {win_ats1.sum()} -> msdiffaug_win_ats1_cases.csv")
    print(f"所有模型均失败(hit1=0): {all_fail.sum()} -> all_models_fail_cases.csv")
    print(f"其他模型全成功而msdiffaug失败: {msdiffaug_fail_others_win.sum()} "
          f"-> msdiffaug_fail_others_win_cases.csv")
    print(f"msdiffaug 失败(hit1=0): {msdiffaug_fail.sum()} -> msdiffaug_fail_cases.csv")


if __name__ == "__main__":
    main()
