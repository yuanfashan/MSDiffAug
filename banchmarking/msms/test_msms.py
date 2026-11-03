# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 09:59:38 2023

@author: Administrator
"""
import pickle
import torch
from matchms import Spectrum
from data_process.process_data import MakeTrainData
from utils.ms_utils import ParseOrbitrap, ModelEmbed, SearchTop_MemoryEfficient, ModelEmbed2, SearchTop
from utils.ms_utils import MPR_MemoryEfficien, SearchTop10ATS_MemoryEfficien
import torch
from banchmarking.msms.ms2vec  import ms_to_vec
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.ERROR)
from utils.ms_utils import MassShiftJitter

def tokens_to_spectra(data):
    """MakeTrainData 格式 (input_ids, intensity, res, intensity_bin) -> matchms Spectrum 列表。

    仅还原真实峰 (input_ids != 0, 不含第 0 位前体峰), m/z = (token - 2) / 100,
    按 m/z 升序排列。强度为归一化值, MSMS2Vec 的 transform 内部会按基峰归一,
    且检索为余弦相似度, 对全局强度缩放不变, 不影响结果。
    """
    spectra = []
    for input_ids, intensity, _, _ in data:
        pos = [p for p in range(1, len(input_ids)) if input_ids[p] != 0]
        mz = np.array([(input_ids[p] - 2) / 100.0 for p in pos])
        ints = np.array([intensity[0, p] for p in pos])
        order = np.argsort(mz)
        spectra.append(Spectrum(mz=mz[order], intensities=ints[order], metadata={}))
    return spectra


def MSMS2Vec(spectrums):
    '''
    Converting MS/MS into vectors using binning method
    '''
    m = ms_to_vec()
    peaks = [s.peaks.to_numpy for s in spectrums]
    peaks = [m.transform(i) for i in peaks]
    peaks_vec = np.array(peaks)
    return peaks_vec


def filter_query_data_by_common_smiles(ref_smiles, query_smiles, query_data):
    # 确保 ref_smiles 是一个集合 (set) 以进行高效的成员查找
    # 集合查找的平均时间复杂度为 O(1)，而列表查找为 O(n)
    ref_smiles_set = set(ref_smiles)

    # 检查输入列表长度是否一致
    if len(query_smiles) != len(query_data):
        raise ValueError("query_smiles 和 query_data 的长度必须一致。")

    # 用于存放过滤后的结果
    filtered_query_smiles = []
    filtered_query_data = []

    # 遍历 query_smiles 和 query_data
    for smile, data in zip(query_smiles, query_data):
        # 检查当前 smile 是否包含在 ref_smiles 集合中
        if smile in ref_smiles_set:
            # 如果包含，则保留，并添加到结果列表中
            filtered_query_smiles.append(smile)
            filtered_query_data.append(data)

    return filtered_query_data, filtered_query_smiles

if __name__ == '__main__':
    max_len = 100
    ref, test_msms1, test_precursor1, ref_smiles = ParseOrbitrap(
        '/home/ausu/projects/yfs/data/GNPSdata/ob_test_ref.pickle')
    ref = [s for s1 in ref for s in s1]


    query, query_msms1, query_precursor1, query_smiles = ParseOrbitrap(
         '/home/ausu/projects/yfs/data/GNPSdata/ob_test_query.pickle')
    query = [s for s1 in query for s in s1]

    query_smiles_orig = query_smiles
    query, query_smiles = filter_query_data_by_common_smiles(ref_smiles, query_smiles, query )
    keep_idx = [i for i, s in enumerate(query_smiles_orig) if s in set(ref_smiles)]
    query_token_data, _ = MakeTrainData(query_msms1, query_precursor1, 100)
    query_token_data = [query_token_data[i] for i in keep_idx]
    print(len(query))
    print(len(query_smiles))

    ref_arr = MSMS2Vec(ref)
    query_arr = MSMS2Vec(query)

    top = SearchTop_MemoryEfficient(ref_arr, query_arr, ref_smiles, query_smiles, 1000)  #cpu计算
    MPR = MPR_MemoryEfficien(ref_arr, query_arr, ref_smiles, query_smiles)
    Top10ATS = SearchTop10ATS_MemoryEfficien(ref_arr, query_arr, ref_smiles, query_smiles)
    print('clean MPR:', MPR)
    print('clean top1/top5/top10:', top)
    print('clean Top10ATS:', Top10ATS)


