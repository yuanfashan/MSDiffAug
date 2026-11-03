# -*- coding: utf-8 -*-
"""
统一对比测试脚本: 谱库搜索 + 类似物搜索 (五种嵌入方法共用一套检索与评估协议)

流程:
  1. 按 test_benchmark_config.py 的 MODEL / EMBED_MODE 准备谱库嵌入
     (npz 键名 ref_arr / ref_smiles / ref_ik 与各模型旧脚本一致,
      EMBED_MODE="load" 可直接加载 library/library2/library3 里的旧 npz;
      谱库与查询 npz 都会保存)
  2. 统一余弦检索: 分块 top-K (与 utils.ms_utils 的 SearchTop*_MemoryEfficient
     相同算法, 数字应与旧脚本一致)
  3. TASK 含 "library": 谱库搜索 -> TOP1/5/10 命中率 (InChIKey14 或 SMILES,
     由 HIT_MODE 决定) + ATS1/5/10 (top-k 候选真实 Morgan/Tanimoto 相似度),
     逐查询 CSV + 汇总 CSV
  4. TASK 含 "analog": 类似物搜索 -> 全部查询的 ATS1/5/10 + RECALLS 各覆盖率
     档位下的 ATS1/5/10, 逐查询 CSV (含 query_smiles/top1_smiles/emb_sim/
     tanimoto 列, 可直接被 plot_seach_anolog_gnps_2.py 读取) + 汇总 CSV

用法: 先改 test_benchmark_config.py, 再运行
    python test_benchmark.py
"""
import csv
import os
import pickle
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
for p in (_HERE, _REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
import torch.nn.functional as F

import test_benchmark_config as C

CSV_PREFIX = C.OUT_TAG or C.MODEL


# ================================================================ 嵌入部分

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def flatten_spectra(data):
    """分组 pickle (list[list[Spectrum]]) 展平; 已是扁平列表直接返回。"""
    if data and isinstance(data[0], (list, tuple)):
        return [s for grp in data for s in grp]
    return list(data)


def get_ik14(spec):
    for key in ("inchikey14", "inchikey", "inchi_key", "InChIKey", "inchikey_inchi"):
        v = spec.get(key)
        if isinstance(v, str) and len(v) >= 14:
            return v[:14]
    return None


def save_npz(path, arr, smiles, ik):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path,
             ref_arr=np.asarray(arr),
             ref_smiles=np.array(smiles, dtype=object),
             ref_ik=np.array(ik, dtype=object))
    print(f"  [嵌入] 已保存 {path}: {len(arr)} 条谱")


def load_npz(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"找不到嵌入文件: {path}\n"
            f"  -> 把 EMBED_MODE 改成 \"from-scratch\" 重新生成, "
            f"或把 LIB_NPZ 指向已有 npz")
    data = np.load(path, allow_pickle=True)
    return data["ref_arr"], list(data["ref_smiles"]), list(data["ref_ik"])


def get_device():
    if C.DEVICE == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _slice_queries(spectra_like_list, limit):
    """只对查询集生效的截断 (MAX_QUERIES 前 N 条)。"""
    if limit and C.MAX_QUERIES and C.MAX_QUERIES < len(spectra_like_list):
        return spectra_like_list[:C.MAX_QUERIES]
    return spectra_like_list


def embed_transformer(pickle_path, limit=False):
    """MSDiffAug / MSBERT: ParseOrbitrap_ik -> MakeTrainData -> ModelEmbed。"""
    from data_process.process_data import MakeTrainData
    from utils.ms_utils import ModelEmbed, ParseOrbitrap_ik
    if C.MODEL == "msdiffaug":
        from model.representation.msdiffaug import MSDiffAug
        model = MSDiffAug(100002, 512, 6, 16, 0.1, 100, 3)  # 构造参数与旧脚本一致
    elif C.MODEL == "msdiffaug_pr":
        from model.representation.msdiffaug import MSDiffAug_MaskTop5PlusRandom
        model = MSDiffAug_MaskTop5PlusRandom(100002, 512, 6, 16, 0.1, 100, 3)  # 与 ablition2/test.py 一致
    else:
        _msbert_dir = os.path.join(_HERE, "msbert")
        if _msbert_dir not in sys.path:
            sys.path.insert(0, _msbert_dir)
        from banchmarking.msbert.msbert import MSBERT
        model = MSBERT(100002, 512, 6, 16, 0, 100, 3)
    model.load_state_dict(
        torch.load(C.MODEL_FILE, map_location="cpu", weights_only=False))
    msms, precursor, smiles, ik = ParseOrbitrap_ik(pickle_path)
    if limit and C.MAX_QUERIES and C.MAX_QUERIES < len(msms):
        msms, precursor, smiles, ik = (x[:C.MAX_QUERIES]
                                       for x in (msms, precursor, smiles, ik))
    data, _ = MakeTrainData(msms, precursor, C.MAX_LEN)
    arr = ModelEmbed(model, data, C.EMBED_BATCH)
    return arr, smiles, ik


def embed_ms2deepscore(pickle_path, limit=False):
    """MS2DeepScore: 展平 pickle -> 官方嵌入 (与 ms2deepscore/spectrum_library.py 一致)。"""
    from banchmarking.ms2deepscore import MS2DeepScore
    from banchmarking.ms2deepscore.models import load_model
    model_path = C.MODEL_FILE
    if not os.path.exists(model_path):
        candidate = os.path.join(_HERE, "ms2deepscore", model_path)
        if os.path.exists(candidate):
            model_path = candidate
    model = load_model(model_path)
    spectra = flatten_spectra(load_pickle(pickle_path))
    spectra = _slice_queries(spectra, limit)
    scorer = MS2DeepScore(model)
    if get_device() == "cuda":
        arr = scorer.get_embedding_array_gpu(spectra, batch_size=C.MS2DS_EMBED_BATCH)
    else:
        arr = scorer.get_embedding_array(spectra)
    smiles = [s.get("smiles") for s in spectra]
    ik = [get_ik14(s) for s in spectra]
    return np.asarray(arr), smiles, ik


def embed_spec2vec(pickle_path, limit=False):
    """Spec2Vec: 展平 + 前体过滤 -> SpectrumDocument -> calc_vector (与旧脚本一致)。"""
    import gensim
    from spec2vec import SpectrumDocument
    _spe2vec_dir = os.path.join(_HERE, "spe2vec")
    if _spe2vec_dir not in sys.path:
        sys.path.insert(0, _spe2vec_dir)
    from banchmarking.spe2vec.spec2wordvector import calc_vector
    spectra = flatten_spectra(load_pickle(pickle_path))
    spectra = [s for s in spectra
               if s.metadata.get("precursor_mz") is not None
               and 0.0 < float(s.metadata["precursor_mz"]) < 1000.0]
    spectra = _slice_queries(spectra, limit)
    model = gensim.models.Word2Vec.load(C.MODEL_FILE)
    arr = []
    for i in range(0, len(spectra), 100):
        for s in spectra[i:i + 100]:
            v, _ = calc_vector(
                model, SpectrumDocument(s, n_decimals=C.SPEC2VEC_N_DECIMALS),
                intensity_weighting_power=C.SPEC2VEC_INTENSITY_POWER,
                allowed_missing_percentage=C.SPEC2VEC_ALLOWED_MISSING)
            arr.append(v)
    smiles = [s.get("smiles") for s in spectra]
    ik = [get_ik14(s) for s in spectra]
    return np.asarray(arr), smiles, ik


def embed_msms(pickle_path, limit=False):
    """Binned MS/MS: 展平 pickle -> 1 Da 分箱向量 (与 msms/spectrum_library.py 一致)。"""
    from banchmarking.msms.ms2vec import ms_to_vec
    spectra = flatten_spectra(load_pickle(pickle_path))
    spectra = _slice_queries(spectra, limit)
    m = ms_to_vec()
    arr = np.array([m.transform(s.peaks.to_numpy) for s in spectra])
    smiles = [s.get("smiles") for s in spectra]
    ik = [get_ik14(s) for s in spectra]
    return arr, smiles, ik


def embed_data(pickle_path, limit=False):
    print(f"[嵌入] {C.MODEL} <- {pickle_path}")
    t0 = time.time()
    if C.MODEL in ("msdiffaug", "msdiffaug_pr", "msbert"):
        arr, smiles, ik = embed_transformer(pickle_path, limit)
    elif C.MODEL == "ms2deepscore":
        arr, smiles, ik = embed_ms2deepscore(pickle_path, limit)
    elif C.MODEL == "spec2vec":
        arr, smiles, ik = embed_spec2vec(pickle_path, limit)
    elif C.MODEL == "msms":
        arr, smiles, ik = embed_msms(pickle_path, limit)
    else:
        raise ValueError(f"未知 MODEL: {C.MODEL}")
    assert len(arr) == len(smiles) == len(ik), \
        f"嵌入结果长度不一致: {len(arr)}/{len(smiles)}/{len(ik)}"
    print(f"  [嵌入] 完成 {len(arr)} 条, 耗时 {time.time() - t0:.1f}s")
    return arr, smiles, ik


# ================================================================ 检索部分

def retrieve_topk(ref_arr, query_arr, top_k=C.TOP_K, batch=C.RETRIEVAL_BATCH,
                  chunk=C.RETRIEVAL_CHUNK, device=None):
    """分块余弦检索, 返回每查询的 top-K 分数与谱库索引 (与旧脚本同算法)。"""
    device = device or get_device()
    n_lib, n_q = ref_arr.shape[0], query_arr.shape[0]
    best_scores = np.full((n_q, top_k), -np.inf, dtype=np.float32)
    best_indices = np.full((n_q, top_k), -1, dtype=np.int64)
    ref_t = torch.from_numpy(ref_arr.astype(np.float32))
    q_t = torch.from_numpy(query_arr.astype(np.float32))
    with torch.no_grad():
        for s in range(0, n_lib, chunk):
            e = min(s + chunk, n_lib)
            rc = F.normalize(ref_t[s:e].to(device), dim=1)
            chunk_idx = np.arange(s, e)
            for qs in range(0, n_q, batch):
                qe = min(qs + batch, n_q)
                qb = F.normalize(q_t[qs:qe].to(device), dim=1)
                scores = (qb @ rc.t()).cpu().numpy()
                for i in range(scores.shape[0]):
                    qi = qs + i
                    merged_s = np.concatenate([best_scores[qi], scores[i]])
                    merged_i = np.concatenate([best_indices[qi], chunk_idx])
                    if merged_s.size > top_k:
                        top = np.argpartition(merged_s, -top_k)[-top_k:]
                    else:
                        top = np.arange(merged_s.size)
                    order = np.argsort(merged_s[top])[::-1]
                    best_scores[qi] = merged_s[top[order]]
                    best_indices[qi] = merged_i[top[order]]
    return best_scores, best_indices


def make_tanimoto_fn(radius=C.FP_RADIUS, bits=C.FP_BITS):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import DataStructs, rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.error")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)
    fp_cache = {}

    def get_fp(smi):
        if not smi:
            return None
        if smi not in fp_cache:
            mol = Chem.MolFromSmiles(str(smi))
            fp_cache[smi] = gen.GetFingerprint(mol) if mol is not None else None
        return fp_cache[smi]

    def tani(sa, sb):
        fa, fb = get_fp(sa), get_fp(sb)
        if fa is None or fb is None:
            return np.nan
        return DataStructs.TanimotoSimilarity(fa, fb)

    return tani, get_fp


def _per_query_ats(tani, q_smiles, ref_smiles, cand_idx):
    """逐查询 ATS1/5/10: top-k 候选里可解析结构的 Tanimoto 均值 (与旧脚本一致)。"""
    tans = [tani(q_smiles, ref_smiles[c]) for c in cand_idx]
    tans = [t for t in tans if not np.isnan(t)]
    return {k: (float(np.mean(tans[:k])) if tans else float("nan"))
            for k in (1, 5, 10)}


def _write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [保存] {path} ({len(rows)} 行)")


# ================================================================ 谱库搜索

def run_library_search(ref_arr, ref_smiles, ref_ik,
                       query_arr, query_smiles, query_ik, out_dir):
    print("\n[谱库搜索] TOP1/5/10 + ATS1/5/10 ...")
    t0 = time.time()
    tani, _ = make_tanimoto_fn()
    best_scores, best_indices = retrieve_topk(ref_arr, query_arr)
    n_q = query_arr.shape[0]
    ik_mode = (C.HIT_MODE == "inchikey")
    if ik_mode:
        n_ik = sum(1 for x in query_ik if x)
        print(f"  InChIKey14 覆盖: 查询 {n_ik}/{n_q}"
              f"{'  [警告: 覆盖为0, 换 HIT_MODE=smiles]' if n_ik == 0 else ''}")

    def is_hit(qi, ci):
        if ik_mode:
            return (query_ik[qi] is not None and ref_ik[ci] is not None
                    and query_ik[qi] == ref_ik[ci])
        return query_smiles[qi] == ref_smiles[ci]

    hit_counts = [0, 0, 0]
    ats_agg = {k: [] for k in (1, 5, 10)}
    rows = []
    for i in range(n_q):
        cand = best_indices[i][best_indices[i] >= 0]
        top1_smiles = ref_smiles[int(cand[0])] if cand.size else ""
        top1_ik = ref_ik[int(cand[0])] if cand.size else ""
        top1_sim = float(best_scores[i, 0]) if cand.size else float("nan")
        hits = [is_hit(i, int(c)) for c in cand] if cand.size else []
        h1 = bool(hits[0]) if hits else False
        h5 = any(hits[:5]) if hits else False
        h10 = any(hits[:10]) if hits else False
        hit_counts[0] += h1
        hit_counts[1] += h5
        hit_counts[2] += h10
        ats = _per_query_ats(tani, query_smiles[i], ref_smiles,
                             [int(c) for c in cand])
        for k in (1, 5, 10):
            if not np.isnan(ats[k]):
                ats_agg[k].append(ats[k])
        rows.append({
            "query_idx": i,
            "query_smiles": query_smiles[i] or "",
            "query_ik": query_ik[i] or "",
            "top1_smiles": top1_smiles,
            "top1_ik": top1_ik or "",
            "top1_emb_sim": top1_sim,
            "hit1": int(h1), "hit5": int(h5), "hit10": int(h10),
            "ats1": ats[1], "ats5": ats[5], "ats10": ats[10],
        })
    top = [c / n_q for c in hit_counts]
    ats_mean = {k: (float(np.mean(ats_agg[k])) if ats_agg[k] else float("nan"))
                for k in (1, 5, 10)}
    _write_csv(os.path.join(out_dir, f"{CSV_PREFIX}_libsearch_per_query.csv"),
               ["query_idx", "query_smiles", "query_ik", "top1_smiles",
                "top1_ik", "top1_emb_sim", "hit1", "hit5", "hit10",
                "ats1", "ats5", "ats10"], rows)
    _write_csv(os.path.join(out_dir, f"{CSV_PREFIX}_libsearch_summary.csv"),
               ["metric", "value"],
               [{"metric": "top1", "value": top[0]},
                {"metric": "top5", "value": top[1]},
                {"metric": "top10", "value": top[2]},
                {"metric": "ATS1", "value": ats_mean[1]},
                {"metric": "ATS5", "value": ats_mean[5]},
                {"metric": "ATS10", "value": ats_mean[10]},
                {"metric": "n_queries", "value": n_q}])
    print(f"  top1/top5/top10: {top[0]:.4f} / {top[1]:.4f} / {top[2]:.4f}")
    print(f"  ATS1/ATS5/ATS10: {ats_mean[1]:.4f} / {ats_mean[5]:.4f} / "
          f"{ats_mean[10]:.4f}  (耗时 {time.time() - t0:.1f}s)")


# ================================================================ 类似物搜索

def run_analog_search(ref_arr, ref_smiles, query_arr, query_smiles, out_dir):
    print("\n[类似物搜索] ATS1/5/10 (全部查询 + 各 recall 档位) ...")
    t0 = time.time()
    tani, get_fp = make_tanimoto_fn()
    best_scores, best_indices = retrieve_topk(ref_arr, query_arr)
    n_q = query_arr.shape[0]

    # 逐查询: top1 嵌入相似度与真实相似度 + ATS1/5/10
    rows = []
    valid_idx = []
    ats_agg = {k: [] for k in (1, 5, 10)}
    for i in range(n_q):
        cand = best_indices[i][best_indices[i] >= 0]
        top1_smiles = ref_smiles[int(cand[0])] if cand.size else ""
        top1_sim = float(best_scores[i, 0]) if cand.size else float("nan")
        top1_tani = (float(tani(query_smiles[i], top1_smiles))
                     if top1_smiles else float("nan"))
        ats = _per_query_ats(tani, query_smiles[i], ref_smiles,
                             [int(c) for c in cand])
        rows.append({"query_smiles": query_smiles[i] or "",
                     "top1_smiles": top1_smiles,
                     "emb_sim": top1_sim,
                     "tanimoto": top1_tani,
                     "ats1": ats[1], "ats5": ats[5], "ats10": ats[10]})
        if get_fp(query_smiles[i]) is not None and cand.size > 0:
            valid_idx.append(i)
            for k in (1, 5, 10):
                if not np.isnan(ats[k]):
                    ats_agg[k].append(ats[k])
    _write_csv(os.path.join(out_dir, f"{CSV_PREFIX}_analog_per_query.csv"),
               ["query_smiles", "top1_smiles", "emb_sim", "tanimoto",
                "ats1", "ats5", "ats10"], rows)

    summary = [{"recall": "all", "threshold": "",
                "n_valid": len(valid_idx), "n_covered": len(valid_idx),
                "ATS1": float(np.mean(ats_agg[1])) if ats_agg[1] else float("nan"),
                "ATS5": float(np.mean(ats_agg[5])) if ats_agg[5] else float("nan"),
                "ATS10": float(np.mean(ats_agg[10])) if ats_agg[10] else float("nan")}]
    print(f"  全部查询 (n_valid={len(valid_idx)}): "
          f"ATS1={summary[0]['ATS1']:.4f} ATS5={summary[0]['ATS5']:.4f} "
          f"ATS10={summary[0]['ATS10']:.4f}")

    # 各 recall 档位: 覆盖率 = top1 嵌入相似度 >= 分位数阈值的查询占比
    top1_sims = best_scores[valid_idx, 0]
    if top1_sims.size == 0:
        print("  [警告] 没有有效查询, 跳过 recall 档位")
        return
    for r in C.RECALLS:
        t = float(np.percentile(top1_sims, 100.0 * (1.0 - r)))
        vals = {k: [] for k in (1, 5, 10)}
        covered = []
        for i in valid_idx:
            if best_scores[i, 0] < t:
                continue
            covered.append(i)
            cand = best_indices[i][best_indices[i] >= 0]
            ats = _per_query_ats(tani, query_smiles[i], ref_smiles,
                                 [int(c) for c in cand])
            for k in (1, 5, 10):
                if not np.isnan(ats[k]):
                    vals[k].append(ats[k])
        summary.append({"recall": r, "threshold": t,
                        "n_valid": len(valid_idx), "n_covered": len(covered),
                        "ATS1": float(np.mean(vals[1])) if vals[1] else float("nan"),
                        "ATS5": float(np.mean(vals[5])) if vals[5] else float("nan"),
                        "ATS10": float(np.mean(vals[10])) if vals[10] else float("nan")})
        print(f"  recall={r:.1f} (覆盖 {len(covered)}/{len(valid_idx)}, "
              f"阈值 {t:.4f}): ATS1={summary[-1]['ATS1']:.4f} "
              f"ATS5={summary[-1]['ATS5']:.4f} ATS10={summary[-1]['ATS10']:.4f}")
    _write_csv(os.path.join(out_dir, f"{CSV_PREFIX}_analog_summary.csv"),
               ["recall", "threshold", "n_valid", "n_covered",
                "ATS1", "ATS5", "ATS10"], summary)
    print(f"  (耗时 {time.time() - t0:.1f}s)")


# ================================================================ 主流程

def main():
    print("=" * 60)
    print(f"统一对比测试: MODEL={C.MODEL}, TASK={C.TASK}, "
          f"EMBED_MODE={C.EMBED_MODE}, DEVICE={get_device()}")
    print("=" * 60)

    out_dir = os.path.join(C.OUT_DIR, C.MODEL)
    os.makedirs(out_dir, exist_ok=True)

    # ---------- 1. 嵌入 ----------
    # 谱库: 从 npz 加载或从头嵌入并保存; 查询: 每次现场嵌入并保存 npz
    if C.EMBED_MODE == "load":
        ref_arr, ref_smiles, ref_ik = load_npz(C.LIB_NPZ)
    else:  # from-scratch
        ref_arr, ref_smiles, ref_ik = embed_data(C.LIBRARY_DATA, limit=False)
        save_npz(C.LIB_NPZ, ref_arr, ref_smiles, ref_ik)
    query_arr, query_smiles, query_ik = embed_data(C.QUERY_DATA, limit=True)
    save_npz(C.QUERY_NPZ, query_arr, query_smiles, query_ik)

    print(f"[检索] 谱库 {ref_arr.shape[0]} 条, 查询 {query_arr.shape[0]} 条, "
          f"嵌入维度 {ref_arr.shape[1]}")

    # ---------- 2. 评估 ----------
    if C.TASK in ("library", "both"):
        run_library_search(ref_arr, ref_smiles, ref_ik,
                           query_arr, query_smiles, query_ik, out_dir)
    if C.TASK in ("analog", "both"):
        run_analog_search(ref_arr, ref_smiles, query_arr, query_smiles, out_dir)

    print(f"\n[*] 全部完成, 结果在 {out_dir}")


if __name__ == "__main__":
    main()
