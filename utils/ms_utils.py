# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 16:58:02 2023

@author: Administrator
"""
import torch
from data_process.dataset import MyDataSet, MyDataSet3
import torch.utils.data as Data
from tqdm import tqdm
import numpy as np
import math
import matplotlib.pyplot as plt
import pickle
from data_process.load_gnps_data import ProDataset, MakeDataset
from data_process.process_data import MakeTestData, MakeTrainData
from banchmarking.msms.ms2vec import ms_to_vec
from matchms.importing import load_from_msp, load_from_mgf, load_from_mzml
from scipy.spatial.distance import cosine


# from draft.Ablation.MLM.model.a6.train import MyDataSet

def CalCosineTop(qtof_ref, qtof_query):
    '''
    Library matching performance for calculating cosine similarity
    '''
    qtof_ref_ = [i for s in qtof_ref for i in s]
    qtof_query_ = [i for s in qtof_query for i in s]
    qtof_ref_peak = [s.peaks.to_numpy for s in qtof_ref_]
    qtof_query_peak = [s.peaks.to_numpy for s in qtof_query_]
    m = ms_to_vec()
    qtof_ref_peak = [m.transform(i) for i in qtof_ref_peak]
    qtof_ref_peak = np.array(qtof_ref_peak)
    qtof_query_peak = [m.transform(i) for i in qtof_query_peak]
    qtof_query_peak = np.array(qtof_query_peak)
    qtof_ref = ProDataset(qtof_ref, 2, 99)
    qtof_query = ProDataset(qtof_query, 2, 99)
    smile_ref = [i[0] for i in qtof_ref]
    smile_query = [i[0] for i in qtof_query]
    cosinetop = SearchTop(qtof_ref_peak, qtof_query_peak, smile_ref, smile_query, batch=50)
    return cosinetop


def ParseOrbitrap(file):
    '''
    Parsing Orbitrap Datasets
    '''
    with open(file, 'rb') as f:
        train_ref = pickle.load(f)
    ref = ProDataset(train_ref, 2, 99)
    msms = [i[2] for i in ref]
    precursor = [i[1] for i in ref]
    smiles = [i[0] for i in ref]
    # parent_mass = [i[3] for i in ref]
    return train_ref, msms, precursor, smiles


def ParseOrbitrap_ik(file):
    '''
    与 ParseOrbitrap 相同, 但返回 ProDataset 处理后的平行列表:
    msms / precursor / smiles / InChIKey前14位 (缺失为 None)
    注意: ParseOrbitrap 返回的第一个值是原始 pickle 内容 (按 smiles 分组),
    不能直接逐谱索引, 所以 ik 列表必须从这里取。
    '''
    with open(file, 'rb') as f:
        train_ref = pickle.load(f)
    ref = ProDataset(train_ref, 2, 99)
    msms = [i[2] for i in ref]
    precursor = [i[1] for i in ref]
    smiles = [i[0] for i in ref]
    inchikeys = [i[3] if len(i) > 3 else None for i in ref]
    return msms, precursor, smiles, inchikeys


def ParseOrbitrap_test(file):
    '''
    Parsing Orbitrap Datasets
    '''
    with open(file, 'rb') as f:
        train_ref = pickle.load(f)
    # train_ref = train_ref[1:]
    ref = ProDataset(train_ref, 2, 99)
    msms = [i[2] for i in ref]
    precursor = [i[1] for i in ref]
    smiles = [i[0] for i in ref]
    return train_ref, msms, precursor, smiles


def CalMSBERTTop(MSBERTModel, ref_data, query_data, smile_ref, smile_query):
    '''
    Library matching performance for MSBERT
    '''
    ref_arr = ModelEmbed(MSBERTModel, ref_data, 64)
    query_arr = ModelEmbed(MSBERTModel, query_data, 64)
    top = SearchTop(ref_arr, query_arr, smile_ref, smile_query, batch=50)
    return top


def ParseOtherData(other):
    '''
    Parsing other types of instrument datasets
    '''
    other_ref, other_query, _, _ = MakeDataset(other, n_max=99, test_size=0, n_decimals=2)
    other_ref = ProDataset(other_ref, 2, 99)
    other_query = ProDataset(other_query, 2, 99)
    msms_ref = [i[2] for i in other_ref]
    msms_query = [i[2] for i in other_query]
    smile_ref = [i[0] for i in other_ref]
    smile_query = [i[0] for i in other_query]
    precursor_ref = [i[1] for i in other_ref]
    precursor_query = [i[1] for i in other_query]
    ref_data, word2idx = MakeTrainData(msms_ref, precursor_ref, 100)
    query_data, word2idx = MakeTrainData(msms_query, precursor_query, 100)
    return ref_data, query_data, smile_ref, smile_query


def DatasetSep(input_ids, intensity, val_size=0.1):
    '''
    Split the dataset for training and testing
    '''
    n = len(intensity)
    perm = np.random.permutation(n)
    n_train = int(n * (1 - val_size))
    perm_train = perm[0:n_train]
    perm_val = perm[n_train:]
    input_ids_train = [input_ids[x] for x in perm_train]
    input_ids_val = [input_ids[x] for x in perm_val]
    intensity_train = [intensity[x] for x in perm_train]
    intensity_val = [intensity[x] for x in perm_val]
    return input_ids_train, intensity_train, input_ids_val, intensity_val


def ModelEmbed(model, test_data, batch_size):
    '''
    Using MSBERT for MS/MS embedding
    Parameters
    ----------
    model
        MSBERT MSBertModel.
    test_data : list
        Data for MSBERT.
    batch_size : int

    Returns
    -------
    embed_arr : array
        MSBERT embedding results.

    '''
    input_ids, intensity, res, inten_bin = zip(*test_data)
    input_ids = [torch.LongTensor(i) for i in input_ids]
    intensity = [torch.FloatTensor(i) for i in intensity]
    res = [torch.FloatTensor(i) for i in res]
    inten_bin = [torch.LongTensor(i) for i in inten_bin]
    dataset = MyDataSet3(input_ids, intensity, res)
    dataloader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    embed_list = []
    with torch.no_grad():
        for step, (input_id, intensity_, res) in tqdm(enumerate(dataloader)):
            input_id = input_id.to(device)
            intensity_ = intensity_.to(device)
            res = res.to(device)
            # pool = model.predict(input_id,intensity_,res)
            # msbert
            pool = model.predict(input_id, intensity_)

            embed_list.append(pool.cpu().numpy())
    embed_arr = np.concatenate(embed_list)
    embed_arr = embed_arr.reshape(embed_arr.shape[0], embed_arr.shape[2])
    return embed_arr


def ModelEmbed_projection(msencoder, projection, test_data, batch_size):
    '''
    Using MSBERT for MS/MS embedding
    Parameters
    ----------
    model
        MSBERT MSBertModel.
    test_data : list
        Data for MSBERT.
    batch_size : int

    Returns
    -------
    embed_arr : array
        MSBERT embedding results.

    '''
    input_ids, intensity = zip(*test_data)
    intensity = [torch.FloatTensor(i) for i in intensity]
    input_ids = [torch.LongTensor(i) for i in input_ids]
    dataset = MyDataSet(input_ids, intensity)
    dataloader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    msencoder = msencoder.to(device)
    msencoder.eval()
    projection = projection.to(device)
    projection.eval()
    embed_list = []
    with torch.no_grad():
        for step, (input_id, intensity_) in tqdm(enumerate(dataloader)):
            input_id = input_id.to(device)
            intensity_ = intensity_.to(device)
            pool = msencoder.predict(input_id, intensity_)
            pool = projection(pool)
            embed_list.append(pool.cpu().numpy())
    embed_arr = np.concatenate(embed_list)
    embed_arr = embed_arr.reshape(embed_arr.shape[0], embed_arr.shape[2])
    return embed_arr


def ModelEmbed2(model, test_data, batch_size):
    '''
    Using MSBERT for MS/MS embedding
    Parameters
    ----------
    model
        MSBERT MSBertModel.
    test_data : list
        Data for MSBERT.
    batch_size : int

    Returns
    -------
    embed_arr : array
        MSBERT embedding results.

    '''
    input_ids, intensity = zip(*test_data)
    intensity = [torch.FloatTensor(i) for i in intensity]
    input_ids = [torch.LongTensor(i) for i in input_ids]
    dataset = MyDataSet(input_ids, intensity)
    dataloader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    embed_list = []
    with torch.no_grad():
        for step, (input_id, intensity_) in tqdm(enumerate(dataloader)):
            input_id = input_id.to(device)
            intensity_ = intensity_.to(device)
            pool = model.predict(input_id, intensity_)
            embed_list.append(pool.cpu().numpy())
    embed_arr = np.concatenate(embed_list)
    embed_arr = embed_arr.reshape(embed_arr.shape[0], embed_arr.shape[2])
    return embed_arr


def SearchTop(dataset_arr, query_arr, dataset_smiles, query_smiles, batch):
    '''
    Top-n metrics for computing library matching
    '''
    top1 = []
    top5 = []
    top10 = []
    start = 0
    n_dataset = np.linalg.norm(dataset_arr, axis=1)
    n_dataset = n_dataset.reshape(n_dataset.shape[0], 1)
    while start < query_arr.shape[0]:
        end = start + batch
        q_i = query_arr[start:end, :]
        n_q = np.linalg.norm(q_i, axis=1)
        n_q = n_q.reshape(1, n_q.shape[0])
        n_q = np.repeat(n_q, n_dataset.shape[0], axis=0)
        dot = np.dot(dataset_arr, q_i.T)
        n_d = np.repeat(n_dataset, q_i.shape[0], axis=1)
        sim = dot / (n_d * n_q)
        sort = np.argsort(sim, axis=0)
        sort = np.flipud(sort)
        for s in range(sort.shape[1]):
            smi_q = query_smiles[(s + start)]
            smi_dataset = [dataset_smiles[i] for i in sort[0:10, s]]
            if smi_q in smi_dataset:
                top10.append(1)
            smi_dataset = [dataset_smiles[i] for i in sort[0:5, s]]
            if smi_q in smi_dataset:
                top5.append(1)
            smi_dataset = [dataset_smiles[i] for i in sort[0:1, s]]
            if smi_q in smi_dataset:
                top1.append(1)
        start += batch
    top1 = len(top1) / len(query_smiles)
    top5 = len(top5) / len(query_smiles)
    top10 = len(top10) / len(query_smiles)
    return [top1, top5, top10]


def SearchTopInChIKey(dataset_arr, query_arr, dataset_ik, query_ik, batch):
    '''
    Top-n metrics for computing library matching
    命中判定: InChIKey 前14位相等 (忽略立体异构), 任一侧为 None 时不计命中
    '''

    def is_hit(q_idx, cand_idx):
        q_ik = query_ik[q_idx]
        c_ik = dataset_ik[cand_idx]
        return q_ik is not None and c_ik is not None and q_ik == c_ik

    top1 = []
    top5 = []
    top10 = []
    start = 0
    n_dataset = np.linalg.norm(dataset_arr, axis=1)
    n_dataset = n_dataset.reshape(n_dataset.shape[0], 1)
    while start < query_arr.shape[0]:
        end = start + batch
        q_i = query_arr[start:end, :]
        n_q = np.linalg.norm(q_i, axis=1)
        n_q = n_q.reshape(1, n_q.shape[0])
        n_q = np.repeat(n_q, n_dataset.shape[0], axis=0)
        dot = np.dot(dataset_arr, q_i.T)
        n_d = np.repeat(n_dataset, q_i.shape[0], axis=1)
        sim = dot / (n_d * n_q)
        sort = np.argsort(sim, axis=0)
        sort = np.flipud(sort)
        for s in range(sort.shape[1]):
            q_idx = s + start
            if any(is_hit(q_idx, i) for i in sort[0:10, s]):
                top10.append(1)
            if any(is_hit(q_idx, i) for i in sort[0:5, s]):
                top5.append(1)
            if any(is_hit(q_idx, i) for i in sort[0:1, s]):
                top1.append(1)
        start += batch
    top1 = len(top1) / len(query_ik)
    top5 = len(top5) / len(query_ik)
    top10 = len(top10) / len(query_ik)
    return [top1, top5, top10]


import torch.nn.functional as F


def SearchTop_MemoryEfficient(dataset_arr, query_arr, dataset_smiles, query_smiles,
                                 batch=500, dataset_chunk_size=5000, device='cuda'):
    """
    解决方案1: Dataset分块 + Query批处理
    适用于: Dataset太大，但Query可以批量处理
    """
    if torch.cuda.is_available() and device == 'cuda':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    torch.cuda.empty_cache()

    n_dataset = len(dataset_arr)
    n_queries = len(query_arr)

    # 为每个query存储最佳匹配
    query_best_scores = np.full((n_queries, 10), -np.inf, dtype=np.float32)
    query_best_indices = np.full((n_queries, 10), -1, dtype=np.int32)

    # print(f"Processing {n_dataset} dataset samples in chunks of {dataset_chunk_size}")

    with torch.no_grad():
        # 分块处理dataset
        for dataset_start in range(0, n_dataset, dataset_chunk_size):
            dataset_end = min(dataset_start + dataset_chunk_size, n_dataset)
            dataset_chunk = torch.from_numpy(dataset_arr[dataset_start:dataset_end]).float().to(device)
            dataset_chunk = F.normalize(dataset_chunk, dim=1)
            #
            # print(
            #     f"Processing dataset chunk {dataset_start // dataset_chunk_size + 1}/{(n_dataset - 1) // dataset_chunk_size + 1}")

            # 批处理query
            for query_start in range(0, n_queries, batch):
                query_end = min(query_start + batch, n_queries)
                query_batch = torch.from_numpy(query_arr[query_start:query_end]).float().to(device)
                query_batch = F.normalize(query_batch, dim=1)

                # 归一化后用矩阵乘法计算余弦相似度，避免广播出 (batch, chunk, D) 大中间张量
                cosine_sim = torch.mm(query_batch, dataset_chunk.t())  # shape: (batch_size, chunk_size)

                # 更新每个query的最佳匹配
                for i in range(cosine_sim.shape[0]):
                    query_idx = query_start + i
                    current_scores = cosine_sim[i].cpu().numpy()
                    current_indices = np.arange(dataset_start, dataset_end)

                    # 合并当前分数和历史最佳分数
                    all_scores = np.concatenate([query_best_scores[query_idx], current_scores])
                    all_indices = np.concatenate([query_best_indices[query_idx], current_indices])

                    # 获取top-10
                    if len(all_scores) > 10:
                        top10_mask = np.argpartition(all_scores, -10)[-10:]
                        query_best_scores[query_idx] = all_scores[top10_mask]
                        query_best_indices[query_idx] = all_indices[top10_mask]

                        # 排序
                        sort_order = np.argsort(query_best_scores[query_idx])[::-1]
                        query_best_scores[query_idx] = query_best_scores[query_idx][sort_order]
                        query_best_indices[query_idx] = query_best_indices[query_idx][sort_order]

            # 清理GPU内存
            del dataset_chunk
            torch.cuda.empty_cache() if device.type == 'cuda' else None

    # 计算top-k准确率
    top1_count = 0
    top5_count = 0
    top10_count = 0

    for i in range(n_queries):
        query_smile = query_smiles[i]
        best_indices = query_best_indices[i]
        valid_indices = best_indices[best_indices >= 0]  # 过滤无效索引

        if len(valid_indices) > 0:
            # 检查命中
            if query_smile == dataset_smiles[valid_indices[0]]:
                top1_count += 1
            if query_smile in [dataset_smiles[idx] for idx in valid_indices[:5]]:
                top5_count += 1
            if query_smile in [dataset_smiles[idx] for idx in valid_indices[:10]]:
                top10_count += 1

    return [top1_count / n_queries, top5_count / n_queries, top10_count / n_queries]


def SearchTopInChIKey_MemoryEfficient(dataset_arr, query_arr, dataset_ik, query_ik,
                                      batch=500, dataset_chunk_size=5000, device='cuda'):
    """
    与 SearchTop_MemoryEfficient 相同的分块余弦检索,
    但命中判定使用 InChIKey 前14位相等 (忽略立体异构), 任一侧为 None 时不计命中。
    """
    if torch.cuda.is_available() and device == 'cuda':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    torch.cuda.empty_cache()

    n_dataset = len(dataset_arr)
    n_queries = len(query_arr)

    # 为每个query存储最佳匹配
    query_best_scores = np.full((n_queries, 10), -np.inf, dtype=np.float32)
    query_best_indices = np.full((n_queries, 10), -1, dtype=np.int32)

    with torch.no_grad():
        # 分块处理dataset
        for dataset_start in range(0, n_dataset, dataset_chunk_size):
            dataset_end = min(dataset_start + dataset_chunk_size, n_dataset)
            dataset_chunk = torch.from_numpy(dataset_arr[dataset_start:dataset_end]).float().to(device)
            dataset_chunk = F.normalize(dataset_chunk, dim=1)

            # 批处理query
            for query_start in range(0, n_queries, batch):
                query_end = min(query_start + batch, n_queries)
                query_batch = torch.from_numpy(query_arr[query_start:query_end]).float().to(device)
                query_batch = F.normalize(query_batch, dim=1)

                # 归一化后用矩阵乘法计算余弦相似度，避免广播出 (batch, chunk, D) 大中间张量
                cosine_sim = torch.mm(query_batch, dataset_chunk.t())  # shape: (batch_size, chunk_size)

                # 更新每个query的最佳匹配
                for i in range(cosine_sim.shape[0]):
                    query_idx = query_start + i
                    current_scores = cosine_sim[i].cpu().numpy()
                    current_indices = np.arange(dataset_start, dataset_end)

                    # 合并当前分数和历史最佳分数
                    all_scores = np.concatenate([query_best_scores[query_idx], current_scores])
                    all_indices = np.concatenate([query_best_indices[query_idx], current_indices])

                    # 获取top-10
                    if len(all_scores) > 10:
                        top10_mask = np.argpartition(all_scores, -10)[-10:]
                        query_best_scores[query_idx] = all_scores[top10_mask]
                        query_best_indices[query_idx] = all_indices[top10_mask]

                        # 排序
                        sort_order = np.argsort(query_best_scores[query_idx])[::-1]
                        query_best_scores[query_idx] = query_best_scores[query_idx][sort_order]
                        query_best_indices[query_idx] = query_best_indices[query_idx][sort_order]

            # 清理GPU内存
            del dataset_chunk
            torch.cuda.empty_cache() if device.type == 'cuda' else None

    def is_hit(q_idx, cand_idx):
        q_ik = query_ik[q_idx]
        c_ik = dataset_ik[cand_idx]
        return q_ik is not None and c_ik is not None and q_ik == c_ik

    # 计算top-k准确率
    top1_count = 0
    top5_count = 0
    top10_count = 0

    for i in range(n_queries):
        best_indices = query_best_indices[i]
        valid_indices = best_indices[best_indices >= 0]  # 过滤无效索引

        if len(valid_indices) > 0:
            # 检查命中
            if is_hit(i, valid_indices[0]):
                top1_count += 1
            if any(is_hit(i, idx) for idx in valid_indices[:5]):
                top5_count += 1
            if any(is_hit(i, idx) for idx in valid_indices[:10]):
                top10_count += 1

    return [top1_count / n_queries, top5_count / n_queries, top10_count / n_queries]


def SearchTopATS_MemoryEfficient(dataset_arr, query_arr, dataset_smiles, query_smiles,
                                 batch=500, dataset_chunk_size=5000, device='cuda',
                                 fp_radius=2, fp_bits=2048):
    """Top-1 类似物检索 (structural analog search) 评估。

    对每个 query 嵌入, 在谱库 (dataset) 嵌入中分块计算余弦最近邻 (top-1),
    并计算 query 与命中谱图的真实结构相似度 (Morgan/Tanimoto)。

    返回 (Ats, top_spectrums):
      Ats: (平均嵌入相似度, 平均真实 Tanimoto)
      top_spectrums: [{query_smiles, top1_smiles, emb_sim, tanimoto}, ...]
    """
    from rdkit import Chem
    from rdkit import RDLogger
    from rdkit.Chem import DataStructs, rdFingerprintGenerator
    RDLogger.DisableLog('rdApp.error')  # 数据中混有非 SMILES 编号, 屏蔽 RDKit 解析报错刷屏

    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=fp_radius, fpSize=fp_bits)

    if torch.cuda.is_available() and device == 'cuda':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    torch.cuda.empty_cache()

    n_dataset = len(dataset_arr)
    n_queries = len(query_arr)

    query_best_scores = np.full(n_queries, -np.inf, dtype=np.float32)
    query_best_indices = np.full(n_queries, -1, dtype=np.int64)

    # print(f"Processing {n_dataset} dataset samples in chunks of {dataset_chunk_size}")
    with torch.no_grad():
        for dataset_start in range(0, n_dataset, dataset_chunk_size):
            dataset_end = min(dataset_start + dataset_chunk_size, n_dataset)
            dataset_chunk = torch.from_numpy(
                dataset_arr[dataset_start:dataset_end]).float().to(device)
            dataset_chunk = F.normalize(dataset_chunk, dim=1)

            for query_start in range(0, n_queries, batch):
                query_end = min(query_start + batch, n_queries)
                query_batch = torch.from_numpy(
                    query_arr[query_start:query_end]).float().to(device)
                query_batch = F.normalize(query_batch, dim=1)

                scores = torch.mm(query_batch, dataset_chunk.t()).cpu().numpy()
                top_idx_local = np.argmax(scores, axis=1)
                top_score = scores[np.arange(scores.shape[0]), top_idx_local]

                q_idx = np.arange(query_start, query_end)
                better = top_score > query_best_scores[q_idx]
                query_best_scores[q_idx[better]] = top_score[better]
                query_best_indices[q_idx[better]] = dataset_start + top_idx_local[better]

            del dataset_chunk
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    # 真实结构相似度: Morgan/Tanimoto, 按 smiles 对缓存避免重复计算
    tani_cache = {}

    def true_tanimoto(smi1, smi2):
        if not smi1 or not smi2:
            return float("nan")
        key = (smi1, smi2)
        if key not in tani_cache:
            mol1 = Chem.MolFromSmiles(smi1)
            mol2 = Chem.MolFromSmiles(smi2)
            if mol1 is None or mol2 is None:
                tani_cache[key] = float("nan")
            else:
                fp1 = fp_gen.GetFingerprint(mol1)
                fp2 = fp_gen.GetFingerprint(mol2)
                tani_cache[key] = DataStructs.TanimotoSimilarity(fp1, fp2)
        return tani_cache[key]

    top_spectrums = []
    for i in range(n_queries):
        idx = int(query_best_indices[i])
        if idx < 0:
            top_spectrums.append({"query_smiles": query_smiles[i], "top1_smiles": "",
                                  "emb_sim": float("nan"), "tanimoto": float("nan")})
            continue
        top_spectrums.append({
            "query_smiles": query_smiles[i],
            "top1_smiles": dataset_smiles[idx],
            "emb_sim": float(query_best_scores[i]),
            "tanimoto": true_tanimoto(query_smiles[i], dataset_smiles[idx]),
        })

    emb_sims = [r["emb_sim"] for r in top_spectrums if r["top1_smiles"]]
    tanis = [r["tanimoto"] for r in top_spectrums if not np.isnan(r["tanimoto"])]
    Ats = (float(np.mean(emb_sims)) if emb_sims else float("nan"),
           float(np.mean(tanis)) if tanis else float("nan"))
    return Ats, top_spectrums


def MPR_MemoryEfficien(dataset_arr, query_arr, dataset_smiles, query_smiles,
                       batch=500, dataset_chunk_size=5000, device='cuda'):
    """平均倒数排名 (MRR) 评估。

    对每个 query 嵌入, 在谱库 (dataset) 嵌入中分块计算余弦最近邻 (top-10),
    取第一个正确命中 (真实 SMILES 相同) 排名的倒数, 对所有 query 取平均;
    top-10 内未命中贡献 0。

    返回 [mrr]
    """
    if torch.cuda.is_available() and device == 'cuda':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    torch.cuda.empty_cache()

    n_dataset = len(dataset_arr)
    n_queries = len(query_arr)
    if n_queries == 0:
        return [0.0]

    query_best_scores = np.full((n_queries, 10), -np.inf, dtype=np.float32)
    query_best_indices = np.full((n_queries, 10), -1, dtype=np.int64)

    # print(f"Processing {n_dataset} dataset samples in chunks of {dataset_chunk_size}")

    with torch.no_grad():
        for dataset_start in range(0, n_dataset, dataset_chunk_size):
            dataset_end = min(dataset_start + dataset_chunk_size, n_dataset)
            dataset_chunk = torch.from_numpy(dataset_arr[dataset_start:dataset_end]).float().to(device)
            dataset_chunk = F.normalize(dataset_chunk, dim=1)

            for query_start in range(0, n_queries, batch):
                query_end = min(query_start + batch, n_queries)
                query_batch = torch.from_numpy(query_arr[query_start:query_end]).float().to(device)
                query_batch = F.normalize(query_batch, dim=1)

                cosine_sim = torch.mm(query_batch, dataset_chunk.t())

                for i in range(cosine_sim.shape[0]):
                    query_idx = query_start + i
                    current_scores = cosine_sim[i].cpu().numpy()
                    current_indices = np.arange(dataset_start, dataset_end)

                    all_scores = np.concatenate([query_best_scores[query_idx], current_scores])
                    all_indices = np.concatenate([query_best_indices[query_idx], current_indices])

                    if len(all_scores) > 10:
                        top10_mask = np.argpartition(all_scores, -10)[-10:]
                    else:
                        top10_mask = np.arange(len(all_scores))
                    merged_scores = all_scores[top10_mask]
                    merged_indices = all_indices[top10_mask]

                    sort_order = np.argsort(merged_scores)[::-1]
                    query_best_scores[query_idx][:len(sort_order)] = merged_scores[sort_order]
                    query_best_indices[query_idx][:len(sort_order)] = merged_indices[sort_order]

            del dataset_chunk
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    mrr_sum = 0.0
    for i in range(n_queries):
        query_smile = query_smiles[i]
        valid_indices = query_best_indices[i][query_best_indices[i] >= 0]
        for rank0, idx in enumerate(valid_indices):
            if query_smile == dataset_smiles[idx]:
                mrr_sum += 1.0 / (rank0 + 1)
                break

    return [mrr_sum / n_queries]


def SearchTop10ATS_MemoryEfficien(dataset_arr, query_arr, dataset_smiles, query_smiles,
                                  batch=500, dataset_chunk_size=5000, device='cuda',
                                  fp_radius=2, fp_bits=2048):
    """Top-10 类似物检索评估: 检索结果与真实化合物的平均最大 Tanimoto 相似度。

    对每个 query 嵌入, 在谱库 (dataset) 嵌入中分块计算余弦最近邻 (top-10),
    计算真实化合物 SMILES 与 top-10 命中结构的 Morgan 指纹 Tanimoto 相似度,
    每个 query 取最大值, 再对所有 query 取平均 (无效 SMILES / 无有效命中不计入)。

    返回 [avg_max_tanimoto]
    """
    from rdkit import Chem
    from rdkit import RDLogger
    from rdkit.Chem import DataStructs, rdFingerprintGenerator
    RDLogger.DisableLog('rdApp.error')  # 数据中混有非 SMILES 编号, 屏蔽 RDKit 解析报错刷屏

    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=fp_radius, fpSize=fp_bits)

    if torch.cuda.is_available() and device == 'cuda':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    torch.cuda.empty_cache()

    n_dataset = len(dataset_arr)
    n_queries = len(query_arr)
    if n_queries == 0:
        return [float("nan")]

    query_best_scores = np.full((n_queries, 10), -np.inf, dtype=np.float32)
    query_best_indices = np.full((n_queries, 10), -1, dtype=np.int64)

    # print(f"Processing {n_dataset} dataset samples in chunks of {dataset_chunk_size}")

    with torch.no_grad():
        for dataset_start in range(0, n_dataset, dataset_chunk_size):
            dataset_end = min(dataset_start + dataset_chunk_size, n_dataset)
            dataset_chunk = torch.from_numpy(dataset_arr[dataset_start:dataset_end]).float().to(device)
            dataset_chunk = F.normalize(dataset_chunk, dim=1)

            for query_start in range(0, n_queries, batch):
                query_end = min(query_start + batch, n_queries)
                query_batch = torch.from_numpy(query_arr[query_start:query_end]).float().to(device)
                query_batch = F.normalize(query_batch, dim=1)

                cosine_sim = torch.mm(query_batch, dataset_chunk.t())

                for i in range(cosine_sim.shape[0]):
                    query_idx = query_start + i
                    current_scores = cosine_sim[i].cpu().numpy()
                    current_indices = np.arange(dataset_start, dataset_end)

                    all_scores = np.concatenate([query_best_scores[query_idx], current_scores])
                    all_indices = np.concatenate([query_best_indices[query_idx], current_indices])

                    if len(all_scores) > 10:
                        top10_mask = np.argpartition(all_scores, -10)[-10:]
                    else:
                        top10_mask = np.arange(len(all_scores))
                    merged_scores = all_scores[top10_mask]
                    merged_indices = all_indices[top10_mask]

                    sort_order = np.argsort(merged_scores)[::-1]
                    query_best_scores[query_idx][:len(sort_order)] = merged_scores[sort_order]
                    query_best_indices[query_idx][:len(sort_order)] = merged_indices[sort_order]

            del dataset_chunk
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    tani_cache = {}

    def tanimoto(smi1, smi2):
        if not smi1 or not smi2:
            return float("nan")
        key = (smi1, smi2)
        if key not in tani_cache:
            mol1 = Chem.MolFromSmiles(smi1)
            mol2 = Chem.MolFromSmiles(smi2)
            if mol1 is None or mol2 is None:
                tani_cache[key] = float("nan")
            else:
                fp1 = fp_gen.GetFingerprint(mol1)
                fp2 = fp_gen.GetFingerprint(mol2)
                tani_cache[key] = DataStructs.TanimotoSimilarity(fp1, fp2)
        return tani_cache[key]

    max_tanis = []
    for i in range(n_queries):
        valid_indices = query_best_indices[i][query_best_indices[i] >= 0]
        if len(valid_indices) == 0:
            continue
        t = [tanimoto(query_smiles[i], dataset_smiles[idx]) for idx in valid_indices]
        t = [x for x in t if not np.isnan(x)]
        if t:
            max_tanis.append(max(t))

    avg_max_tanimoto = float(np.mean(max_tanis)) if max_tanis else float("nan")
    return [avg_max_tanimoto]


def RandomDropPeaks(data, drop_ratio=0.3, seed=None):
    """随机删峰 (peak dropout)。

    对 MakeTrainData 输出的每条谱 (input_ids, intensity, res, intensity_bin),
    随机删除 drop_ratio 比例的真实峰 (不含第 0 位前体峰):
    被删位置 input_ids 置 0 ([PAD]), intensity 和 intensity_bin 同步置 0,
    res 保持不变 (MSDiffAug 嵌入不使用 res)。

    返回删峰后的新列表, 不修改原数据。
    """
    drop_ratio = max(0.0, min(1.0, drop_ratio))
    rng = np.random.default_rng(seed)
    data_new = []
    for sample in data:
        input_ids = list(sample[0])
        intensity = sample[1].copy()
        intensity_bin = sample[3].copy()
        peak_positions = [p for p in range(1, len(input_ids)) if input_ids[p] != 0]
        if peak_positions and drop_ratio > 0:
            num_drop = max(int(len(peak_positions) * drop_ratio), 1)
            drop_positions = rng.choice(peak_positions, num_drop, replace=False)
            for p in drop_positions:
                input_ids[p] = 0
                intensity[0, p] = 0
                intensity_bin[p] = 0
        data_new.append([input_ids, intensity, sample[2], intensity_bin])
    return data_new


def IntensityNoise(data, noise_level=0.5, noise_ratio=1.0, seed=None):
    """乘性强度噪声 (对数正态), 只扰动随机选取的 noise_ratio 比例真实峰。

    被选中峰的强度乘以 exp(noise_level * eps), eps ~ N(0,1):
    noise_level 即强度在 log 尺度的噪声标准差, 模拟质谱强度的对数正态测量误差。
    未选中的峰保持不变; input_ids 与 intensity_bin 不变 (嵌入不使用 intensity_bin)。
    """
    noise_level = max(0.0, noise_level)
    noise_ratio = max(0.0, min(1.0, noise_ratio))
    rng = np.random.default_rng(seed)
    data_new = []
    for sample in data:
        input_ids = list(sample[0])
        intensity = sample[1].copy()
        intensity_bin = sample[3].copy()
        peak_positions = [p for p in range(1, len(input_ids)) if input_ids[p] != 0]
        if peak_positions and noise_level > 0 and noise_ratio > 0:
            num_perturb = min(max(int(round(len(peak_positions) * noise_ratio)), 1),
                              len(peak_positions))
            perturb_positions = rng.choice(peak_positions, num_perturb, replace=False)
            eps = rng.standard_normal(num_perturb)
            for p, e in zip(perturb_positions, eps):
                intensity[0, p] = max(intensity[0, p] * math.exp(noise_level * e), 0.0)
        data_new.append([input_ids, intensity, sample[2], intensity_bin])
    return data_new


def SearchTopATSlist_MemoryEfficien(dataset_arr, query_arr, dataset_smiles, query_smiles,
                                batch=500, dataset_chunk_size=5000, device='cuda',
                                fp_radius=2, fp_bits=2048,
                                topk_list=(10,5,1)):
    """多档位 Top-k 类似物检索评估: ATS@k (top-k 候选平均指纹相似度)。

    对每个 query 嵌入, 在谱库 (dataset) 嵌入中分块计算余弦最近邻, 只保留
    top-K (K = max(topk_list)); 再对每个 k, 计算 query 真实结构与 top-k 命中
    结构的 Morgan 指纹 Tanimoto 相似度, 对每个 query 取 top-k 候选的平均值,
    再对所有 query 取平均 (无效 SMILES / 无有效命中不计入)。

    返回列表, 顺序与降序排列的 topk_list 一致:
      [ATS@500, ATS@250, ATS@100, ATS@50, ATS@25, ATS@10]
    """
    from rdkit import Chem
    from rdkit import RDLogger
    from rdkit.Chem import DataStructs, rdFingerprintGenerator
    import torch.nn.functional as F
    RDLogger.DisableLog('rdApp.error')  # 数据中混有非 SMILES 编号, 屏蔽 RDKit 解析报错刷屏

    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=fp_radius, fpSize=fp_bits)

    if torch.cuda.is_available() and device == 'cuda':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    torch.cuda.empty_cache()

    topk_list = tuple(sorted(set(topk_list), reverse=True))
    K = topk_list[0]

    n_dataset = len(dataset_arr)
    n_queries = len(query_arr)
    if n_queries == 0:
        return [float("nan")] * len(topk_list)

    best_scores = torch.full((n_queries, K), float("-inf"), device=device)
    best_indices = torch.full((n_queries, K), -1, dtype=torch.int64, device=device)

    with torch.no_grad():
        for dataset_start in range(0, n_dataset, dataset_chunk_size):
            dataset_end = min(dataset_start + dataset_chunk_size, n_dataset)
            dataset_chunk = F.normalize(
                torch.from_numpy(dataset_arr[dataset_start:dataset_end]).float().to(device),
                dim=1)

            for query_start in range(0, n_queries, batch):
                query_end = min(query_start + batch, n_queries)
                query_batch = F.normalize(
                    torch.from_numpy(query_arr[query_start:query_end]).float().to(device),
                    dim=1)
                scores = torch.mm(query_batch, dataset_chunk.t())  # (B, chunk)

                q_scores = best_scores[query_start:query_end]
                q_indices = best_indices[query_start:query_end]
                chunk_indices = torch.arange(dataset_start, dataset_end, device=device)
                chunk_indices = chunk_indices.unsqueeze(0).expand(scores.shape[0], -1)

                all_scores = torch.cat([q_scores, scores], dim=1)  # (B, K+chunk)
                all_indices = torch.cat([q_indices, chunk_indices], dim=1)
                top_vals, top_pos = torch.topk(all_scores, k=K, dim=1)
                best_scores[query_start:query_end] = top_vals
                best_indices[query_start:query_end] = all_indices.gather(1, top_pos)

            del dataset_chunk
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    best_indices = best_indices.cpu().numpy()

    fp_cache = {}

    def get_fp(smi):
        if not smi:
            return None
        if smi not in fp_cache:
            mol = Chem.MolFromSmiles(smi)
            fp_cache[smi] = fp_gen.GetFingerprint(mol) if mol is not None else None
        return fp_cache[smi]

    per_query_mean = [[] for _ in topk_list]
    for i in range(n_queries):
        fp_q = get_fp(query_smiles[i])
        if fp_q is None:
            continue
        valid = best_indices[i][best_indices[i] >= 0]
        hit_fps = [get_fp(dataset_smiles[idx]) for idx in valid]
        hit_fps = [fp for fp in hit_fps if fp is not None]
        if not hit_fps:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp_q, hit_fps)
        for j, k in enumerate(topk_list):
            per_query_mean[j].append(float(np.mean(sims[:k])))

    return [float(np.mean(v)) if v else float("nan") for v in per_query_mean]


def SearchTopATSlist_AtRecall(dataset_arr, query_arr, dataset_smiles, query_smiles,
                              recall=0.5, batch=500, dataset_chunk_size=5000,
                              device='cuda', fp_radius=2, fp_bits=2048,
                              topk_list=(10, 5, 1)):
    """指定覆盖率 (recall) 下的 Top-k 类似物检索评估: ATS@k。

    覆盖率定义与 recall-Tanimoto 曲线一致: 以每条有效 query 的 top-1
    嵌入余弦相似度为依据, 阈值 t 取 top-1 相似度的 (1-recall) 分位数
    (recall=0.5 即中位数), top-1 相似度 >= t 的 query 为被覆盖部分;
    再对被覆盖 query 计算 ATS@k (top-k 候选平均 Morgan/Tanimoto)。
    SMILES 无效或无任何命中的 query 不参与分位数与 ATS 计算。

    返回列表, 顺序与降序排列的 topk_list 一致。
    """
    from rdkit import Chem
    from rdkit import RDLogger
    from rdkit.Chem import DataStructs, rdFingerprintGenerator
    import torch.nn.functional as F
    RDLogger.DisableLog('rdApp.error')

    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=fp_radius, fpSize=fp_bits)

    if torch.cuda.is_available() and device == 'cuda':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    torch.cuda.empty_cache()

    topk_list = tuple(sorted(set(topk_list), reverse=True))
    K = topk_list[0]

    n_dataset = len(dataset_arr)
    n_queries = len(query_arr)
    if n_queries == 0:
        return [float("nan")] * len(topk_list)

    best_scores = torch.full((n_queries, K), float("-inf"), device=device)
    best_indices = torch.full((n_queries, K), -1, dtype=torch.int64, device=device)

    with torch.no_grad():
        for dataset_start in range(0, n_dataset, dataset_chunk_size):
            dataset_end = min(dataset_start + dataset_chunk_size, n_dataset)
            dataset_chunk = F.normalize(
                torch.from_numpy(dataset_arr[dataset_start:dataset_end]).float().to(device),
                dim=1)

            for query_start in range(0, n_queries, batch):
                query_end = min(query_start + batch, n_queries)
                query_batch = F.normalize(
                    torch.from_numpy(query_arr[query_start:query_end]).float().to(device),
                    dim=1)
                scores = torch.mm(query_batch, dataset_chunk.t())  # (B, chunk)

                q_scores = best_scores[query_start:query_end]
                q_indices = best_indices[query_start:query_end]
                chunk_indices = torch.arange(dataset_start, dataset_end, device=device)
                chunk_indices = chunk_indices.unsqueeze(0).expand(scores.shape[0], -1)

                all_scores = torch.cat([q_scores, scores], dim=1)  # (B, K+chunk)
                all_indices = torch.cat([q_indices, chunk_indices], dim=1)
                top_vals, top_pos = torch.topk(all_scores, k=K, dim=1)
                best_scores[query_start:query_end] = top_vals
                best_indices[query_start:query_end] = all_indices.gather(1, top_pos)

            del dataset_chunk
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    best_scores = best_scores.cpu().numpy()
    best_indices = best_indices.cpu().numpy()

    fp_cache = {}

    def get_fp(smi):
        if not smi:
            return None
        if smi not in fp_cache:
            mol = Chem.MolFromSmiles(smi)
            fp_cache[smi] = fp_gen.GetFingerprint(mol) if mol is not None else None
        return fp_cache[smi]

    valid_idx = [i for i in range(n_queries)
                 if get_fp(query_smiles[i]) is not None
                 and np.any(best_indices[i] >= 0)]
    if not valid_idx:
        return [float("nan")] * len(topk_list)

    top1_sims = best_scores[valid_idx, 0]
    t = float(np.percentile(top1_sims, 100.0 * (1.0 - recall)))

    per_query_mean = [[] for _ in topk_list]
    for i in valid_idx:
        if best_scores[i, 0] < t:
            continue
        fp_q = get_fp(query_smiles[i])
        valid = best_indices[i][best_indices[i] >= 0]
        hit_fps = [get_fp(dataset_smiles[idx]) for idx in valid]
        hit_fps = [fp for fp in hit_fps if fp is not None]
        if not hit_fps:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp_q, hit_fps)
        for j, k in enumerate(topk_list):
            per_query_mean[j].append(float(np.mean(sims[:k])))

    return [float(np.mean(v)) if v else float("nan") for v in per_query_mean]

def MassShiftJitter(data, jitter_da=0.05, shift_ratio=1.0, seed=None):
    """m/z 抖动 (模拟低质量精度仪器/校准偏差)。

    每条谱随机选择 shift_ratio 比例的真实峰 (不含第 0 位前体峰),
    其 m/z 加均匀扰动 U(-jitter_da, +jitter_da) Da,
    token 偏移 round(delta * 100), 并截断到合法 token 区间 [2, 100001]。
    抖动后落在同一 token 的峰合并 (保留较大强度, 其余置 0);
    与前体峰 m/z 重合的峰并入前体 (前体位置永不清除)。
    返回新列表, 不修改原数据。
    """
    jitter_da = max(0.0, jitter_da)
    shift_ratio = max(0.0, min(1.0, shift_ratio))
    rng = np.random.default_rng(seed)
    data_new = []
    for sample in data:
        input_ids = list(sample[0])
        intensity = sample[1].copy()
        intensity_bin = sample[3].copy()
        peak_positions = [p for p in range(1, len(input_ids)) if input_ids[p] != 0]
        if peak_positions and jitter_da > 0 and shift_ratio > 0:
            num_shift = min(max(int(round(len(peak_positions) * shift_ratio)), 1),
                            len(peak_positions))
            shift_positions = rng.choice(peak_positions, num_shift, replace=False)
            new_ids = list(input_ids)
            shifts = np.round(rng.uniform(-jitter_da, jitter_da, num_shift) * 100).astype(int)
            for p, d in zip(shift_positions, shifts):
                new_ids[p] = int(np.clip(input_ids[p] + d, 2, 100001))
            # 所有真实峰 (含前体) 的 token -> 位置, 用于碰撞合并
            seen = {input_ids[p]: p for p in peak_positions}
            seen[input_ids[0]] = 0
            for p in shift_positions:
                old_tok = input_ids[p]
                tok = new_ids[p]
                if tok == old_tok:
                    continue
                if seen.get(old_tok) == p:
                    del seen[old_tok]
                if tok in seen:
                    q = seen[tok]
                    if q != 0 and intensity[0, p] > intensity[0, q]:
                        new_ids[q] = 0
                        intensity[0, q] = 0.0
                        intensity_bin[q] = 0
                        seen[tok] = p
                    else:
                        new_ids[p] = 0
                        intensity[0, p] = 0.0
                        intensity_bin[p] = 0
                else:
                    seen[tok] = p
            input_ids = new_ids
        data_new.append([input_ids, intensity, sample[2], intensity_bin])
    return data_new


def AddFalsePeaks(data, add_ratio=0.3, seed=None):
    """随机假峰 (chemical noise)。

    在每条谱的空白位置插入 add_ratio * 真实峰数 个假峰:
    m/z 在该谱真实峰 m/z 范围内均匀采样, 强度取 [0.01, 0.05] * 基峰强度,
    与低丰度噪声峰量级一致。空白位置不足时截断; 第 0 位前体峰位置不插入。
    """
    add_ratio = max(0.0, add_ratio)
    rng = np.random.default_rng(seed)
    data_new = []
    for sample in data:
        input_ids = list(sample[0])
        intensity = sample[1].copy()
        intensity_bin = sample[3].copy()
        real_positions = [p for p in range(1, len(input_ids)) if input_ids[p] != 0]
        empty_positions = [p for p in range(1, len(input_ids)) if input_ids[p] == 0]
        if real_positions and empty_positions and add_ratio > 0:
            mzs = [(input_ids[p] - 2) / 100.0 for p in real_positions]
            lo, hi = min(mzs), max(mzs)
            base = max(intensity[0, p] for p in real_positions)
            num_add = min(max(int(round(len(real_positions) * add_ratio)), 1),
                          len(empty_positions))
            for p in rng.choice(empty_positions, num_add, replace=False):
                mz = rng.uniform(lo, hi)
                input_ids[p] = int(np.clip(round(mz * 100) + 2, 2, 100001))
                intensity[0, p] = rng.uniform(0.01, 0.05) * base
                intensity_bin[p] = 0
        data_new.append([input_ids, intensity, sample[2], intensity_bin])
    return data_new


def InjectNoise(data, noise_type, strength=0.3, seed=None):
    """噪声注入统一入口。

    noise_type:
      'drop'        -> RandomDropPeaks (随机删峰),  strength = 删峰比例
      'intensity'   -> IntensityNoise (乘性强度噪声), strength = log 噪声标准差
      'mz_jitter'   -> MassShiftJitter (m/z 抖动),    strength = 抖动幅度 Da
      'false_peaks' -> AddFalsePeaks (随机假峰),      strength = 假峰数/真实峰数
      'clean'       -> 原样返回
    """
    if noise_type == 'drop':
        return RandomDropPeaks(data, drop_ratio=strength, seed=seed)
    if noise_type == 'intensity':
        return IntensityNoise(data, noise_level=strength, seed=seed)
    if noise_type == 'mz_jitter':
        return MassShiftJitter(data, jitter_da=strength, seed=seed)
    if noise_type == 'false_peaks':
        return AddFalsePeaks(data, add_ratio=strength, seed=seed)
    if noise_type == 'clean':
        return data
    raise ValueError(f"未知噪声类型: {noise_type}")


def PlotStepLoss(train_loss, step=100):
    '''
    Draw loss curve
    '''
    all_loss = [p for i in train_loss for p in i]
    step_loss = [all_loss[i:i + step] for i in range(0, len(all_loss), step)]
    step_loss = [np.nanmean(i) for i in step_loss]
    plt.plot(step_loss)
    plt.xlabel('Steps')
    plt.ylabel('Loss')


def ProcessMSP(file):
    '''
    Load dataset from MSP and prepare for MSBERT
    Parameters
    ----------
    file : str
        The Path of MS/MS spectra.
    Returns
    -------
    data : list
        Data for MSBERT.
    smiles : TYPE
        SMILES for MS/MS.
    '''
    p = file.find('.')
    if file[p:] == '.msp':
        msms = list(load_from_msp(file))
    elif file[p:] == '.mgf':
        msms = list(load_from_mgf(file))
    elif file[p:] == '.mzML':
        msms = list(load_from_mzml(file))
    pro_data = ProDataset([msms], 2, 99)
    msms = [i[2] for i in pro_data]
    precursor = [i[1] for i in pro_data]
    smiles = [i[0] for i in pro_data]
    data, _ = MakeTrainData(msms, precursor, 100)
    return data, smiles


def MSBERTSimilarity(r_arr, q_arr):
    '''
    Parameters
    ----------
    r_arr : array
        Reference vectors Embedded by MSBERT.
    q_arr : array
        Query vectors Embedded by MSBERT.

    Returns
    -------
    similarity : arrray
        Similarity matrix.

    '''
    similarity = np.zeros((r_arr.shape[0], q_arr.shape[0]))
    for i in range(r_arr.shape[0]):
        v1 = r_arr[i, :]
        for j in range(q_arr.shape[0]):
            v2 = q_arr[j, :]
            cos_ij = cosine(v1, v2)
            similarity[i, j] = 1 - cos_ij
    return similarity

