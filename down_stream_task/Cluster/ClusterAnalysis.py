# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 21:41:00 2023

@author: Administrator
"""
import pickle
from utils.ms_utils import ModelEmbed
import torch
from ablition5.msbert.msbert import MSBERT
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn import metrics
import numpy as np
from ablition5.spe2vec.TrainSpec2Vec import peak_embed
from data_process.process_data import MakeTrainData
from utils.ms2vec import ms_to_vec
import numpy as np
from data_process.load_gnps_data import ProDataset, MakeDataset
from ablition2.msdiffaug import MSDiffAug_MaskTop5PlusRandom


def ParseOrbitrap(file, index_file=None, min_len=8, n_samples=1000, seed=None, max_per_group=None):
    '''
    Parsing Orbitrap Datasets
    min_len: 只保留长度大于 min_len 的分组
    seed: 随机子集种子; None 时每次运行随机抽不同子集 (多次运行取均值才有意义),
          传具体值可复现某次子集; index_file 仅用于保存本次索引留档
    max_per_group: 每组最多保留的谱数, 超过时随机截断 (控制展平后总谱数, None 不截断)
    '''
    with open(file, 'rb') as f:
        train_ref = pickle.load(f)
    train_ref = [ref for ref in train_ref if len(ref) > min_len]
    if seed is not None:
        np.random.seed(seed)
    indices = np.random.choice(len(train_ref), n_samples, replace=False)
    if index_file:
        np.save(index_file, indices)
    train_ref = [train_ref[i] for i in indices]
    if max_per_group is not None:
        train_ref = [g if len(g) <= max_per_group else
                     [g[j] for j in np.random.choice(len(g), max_per_group, replace=False)]
                     for g in train_ref]
    ref = ProDataset(train_ref, 2, 99)
    msms = [i[2] for i in ref]
    precursor = [i[1] for i in ref]
    smiles = [i[0] for i in ref]
    return train_ref, msms, precursor, smiles


def MSBERTCluster(model_file, test_data, msms, precursor, smiles):
    '''
    Cluster analysis using MSBERT embedded vectors and obtain results
    '''
    MSBERTmodel = MSBERT(100002, 512, 6, 16, 0, 100, 3)
    MSBERTmodel.load_state_dict(torch.load(model_file))
    test_data2, word2idx = MakeTrainData(msms, precursor, 100)
    test_arr = ModelEmbed(MSBERTmodel, test_data2, 32)
    smiles_unique = list(set(smiles))
    labels_true = GenRawLabel(smiles_unique, smiles)
    labels_pred = CosineCluster(len(smiles_unique), test_arr)
    msbert_result = CalEvaluate(labels_true, labels_pred)
    return msbert_result


def MSDiffAugCluster(model_file, test_data, msms, precursor, smiles):
    '''
    Cluster analysis using MSBERT embedded vectors and obtain results
    '''
    MSBERTmodel = MSDiffAug_MaskTop5PlusRandom(100002, 512, 6, 16, 0, 100, 3)
    MSBERTmodel.load_state_dict(torch.load(model_file))
    test_data2, word2idx = MakeTrainData(msms, precursor, 100)
    test_arr = ModelEmbed(MSBERTmodel, test_data2, 32)
    smiles_unique = list(set(smiles))
    labels_true = GenRawLabel(smiles_unique, smiles)
    labels_pred = CosineCluster(len(smiles_unique), test_arr)
    msbert_result = CalEvaluate(labels_true, labels_pred)
    return msbert_result


def Spec2VecCluster(model_file, test_data, smiles):
    '''
    Cluster analysis using Spec2Vec embedded vectors and obtain results
    '''
    smiles_unique = list(set(smiles))
    labels_true = GenRawLabel(smiles_unique, smiles)
    spectrums = [s for s1 in test_data for s in s1]
    spec2vec_vec = peak_embed(model_file, spectrums, n_decimals=2)
    spec2vec_vec = np.array(spec2vec_vec)
    labels_pred = CosineCluster(len(smiles_unique), spec2vec_vec)
    spec2vec_result = CalEvaluate(labels_true, labels_pred)
    return spec2vec_result


def MSMSCluster(test_data, smiles):
    '''
    Cluster analysis using raw MS/MS and obtain results
    '''
    smiles_unique = list(set(smiles))
    labels_true = GenRawLabel(smiles_unique, smiles)
    spectrums = [s for s1 in test_data for s in s1]
    peaks_vec = MSMS2Vec(spectrums)
    labels_pred = CosineCluster(len(smiles_unique), peaks_vec)
    msms_result = CalEvaluate(labels_true, labels_pred)
    return msms_result


def MSMS2Vec(spectrums):
    '''
    Converting MS/MS into vectors using binning method
    '''
    m = ms_to_vec()
    peaks = [s.peaks.to_numpy for s in spectrums]
    peaks = [m.transform(i) for i in peaks]
    peaks_vec = np.array(peaks)
    return peaks_vec


def CosineCluster(n_clusters, test_arr):
    '''
    Cosine agglomerative clustering (scale-invariant, consistent with retrieval metric)
    '''
    # scipy 的 average linkage 用 nearest-neighbor chain 算法, O(n^2);
    # sklearn 的 AgglomerativeClustering 是 O(n^3), 谱库规模下不可用
    if torch.cuda.is_available():
        x = torch.from_numpy(test_arr).float().cuda()
        x = torch.nn.functional.normalize(x, dim=1, eps=1e-8)
        dist = 1.0 - x @ x.T
        condensed = squareform(dist.clamp_min(0).cpu().numpy().astype(np.float64), checks=False)
    else:
        condensed = pdist(test_arr, metric='cosine')
    Z = linkage(condensed, method='average')
    labels_pred = fcluster(Z, t=n_clusters, criterion='maxclust')
    return labels_pred


def GenRawLabel(smiles_unique, smiles):
    '''
    Generate true labels from SMILES
    '''
    smi = smiles_unique[0]
    raw_label = np.zeros(len(smiles))
    l = 0
    for smi in smiles_unique:
        idx = [i for i, v in enumerate(smiles) if v == smi]
        raw_label[idx] = l
        l += 1
    return raw_label


def CalEvaluate(labels_true, labels_pred):
    '''
    Calculate clustering metrics
    '''
    ari = metrics.adjusted_rand_score(labels_true, labels_pred)
    homogeneity = metrics.homogeneity_score(labels_true, labels_pred)
    completeness = metrics.completeness_score(labels_true, labels_pred)
    v_measure = metrics.v_measure_score(labels_true, labels_pred, beta=0.5)
    result = {'ARI': ari, 'homogeneity': homogeneity,
              'completeness': completeness, 'v_measure': v_measure}
    return result


if __name__ == '__main__':
    # MIN_LEN: 分组长度阈值; SEED: 每次运行传不同 seed (如 100/101/102) 以得到不同子集;
    # MAX_PER_GROUP: 每组最多保留的谱数, 控制展平后总谱数, 避免聚类相似度矩阵爆显存
    MIN_LEN = 8
    SEED = 100  # 100, 101，102
    MAX_PER_GROUP = 5
    test_ref, msms3, precursor3, smiles3 = ParseOrbitrap(
        '/home/ausu/projects/yfs/data/GNPSdata_f2/ob_train_ref.pickle',
        min_len=MIN_LEN, seed=SEED, max_per_group=MAX_PER_GROUP)
    #
    # # model_file = '/home/ausu/projects/yfs/MSDiffAugServer/ablition5/ms2deepscore/ms2deepscore_trained11.model'
    #
    # model_file = '/home/ausu/projects/yfs/MSDiffAugServer/ablition2/pth/MaskTop5PlusRandom1/MSDiffAug.pt'
    model_file = '/home/ausu/projects/yfs/MSDiffAugServer/ablition8/p/t100/MSDiffAug_126.pt'
    msbert_result = MSDiffAugCluster(model_file,test_ref,msms3,precursor3,smiles3)
    # 打印评估指标
    print("MSDiffAug 聚类评估结果：")
    print("调整兰德指数 (ARI):", msbert_result['ARI'])
    print("同质性 (Homogeneity):", msbert_result['homogeneity'])
    print("完整性 (Completeness):", msbert_result['completeness'])
    print("V-Measure:", msbert_result['v_measure'])

    model_file = '/home/ausu/projects/yfs/MSDiffAugServer/ablition5//msbert/pth/msbert_12.pt'
    msbert_result = MSBERTCluster(model_file, test_ref, msms3, precursor3, smiles3)
    # 打印评估指标
    print("MSBERT 聚类评估结果：")
    print("调整兰德指数 (ARI):", msbert_result['ARI'])
    print("同质性 (Homogeneity):", msbert_result['homogeneity'])
    print("完整性 (Completeness):", msbert_result['completeness'])
    print("V-Measure:", msbert_result['v_measure'])


    model_file = '/home/ausu/projects/yfs/MSDiffAugServer/ablition5//spe2vec/Spec2VecModel3/ob_spec2vec.model'
    spec2vec_result = Spec2VecCluster(model_file,test_ref,smiles3)
    print("Spec2Vec 聚类评估结果：")
    print("调整兰德指数 (ARI):", spec2vec_result['ARI'])
    print("同质性 (Homogeneity):", spec2vec_result['homogeneity'])
    print("完整性 (Completeness):", spec2vec_result['completeness'])
    print("V-Measure:", spec2vec_result['v_measure'])

    # 调用 MSMSCluster 方法
    msms_result = MSMSCluster(test_ref,smiles3)
    print("MS/MS Bin 聚类评估结果：")
    print("调整兰德指数 (ARI):", msms_result['ARI'])
    print("同质性 (Homogeneity):", msms_result['homogeneity'])
    print("完整性 (Completeness):", msms_result['completeness'])
    print("V-Measure:", msms_result['v_measure'])






