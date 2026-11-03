# -*- coding: utf-8 -*-
"""MSDiffAug 嵌入 t-SNE (新样式): 与 spec2vec_tsne.py / msbert.py 相同的
10 个 SMILES、相同的类别配色, 便于跨模型对比。

- 数据: GNPS ob_train_ref 谱, 前体过滤与 ProDataset 一致
- 嵌入: MSDiffAug_MaskTop5PlusRandom 编码器
  (ablition2/pth/MaskTop5PlusRandom1/MSDiffAug.pt)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 服务器无显示环境
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch

from model.representation.msdiffaug import MSDiffAug_MaskTop5PlusRandom
from utils.ms_utils import ParseOrbitrap, ModelEmbed, MakeTrainData

DATA_PATH = '/home/ausu/projects/yfs/data/GNPSdata_f2/ob_train_ref.pickle'
MODEL_FILE = '/home/ausu/projects/yfs/MSDiffAugServer/ablition2/pth/MaskTop5PlusRandom1/MSDiffAug.pt'
OUT_PNG = 'MSDiffAug.png'
OUT_PDF = 'MSDiffAug..pdf'
MAX_LEN = 100

SELECTED_SMILES = {'O=C(OC1CC2C(O)(CC=C3CC(O)CCC32C)C4(O)CCC(O)(C(=O)C)C14C)C5=CC=C(O)C=C5',
                   'O=C1C=C(OC=2C=C(O)C=C(O)C12)C=3C=CC=CC3',
                   'O=C(OC)C1(O)N2C=3C=CC=CC3C4=C2C5N(CC4)CCCC5(CC)C1',
                   'OCC1OC(OC2C(O)C(O)C(OC2OC3CCC4(C)C(CCC5(C)C4CC(O)C6C(CCC65C)C(O)(C)CCC=C(C)C)C3(C)C)CO)C(O)C(O)C1O',
                   'O=C(O)C12CCC(C)C(C)C2C3=CCC4C5(C)CC(O)C(O)C(C)(CO)C5CCC4(C)C3(C)CC1',
                   'O=C1OC=2C(C=C1)=CC=3C=COC3C2OCC(O)C(O)(C)C',
                   'O=C1C=2C(O)=CC(O)=CC2OC(C3=CC=C4OC(C5=CC=C(OC)C(O)=C5)C(OC4=C3)CO)C1O',
                   'O=C(OC1=C(OC(=O)C)C2=C(C=C1C(C)C)CCC3C2(C(=O)OC)CCCC3(C)C)C',
                   'O=C1CCC2C(C)(CCC3C2(C)CCC4(C)C5CC(C)(C)CCC5(C)CCC34C)C1C',
                   'O=C(O)C=CC1=CC=C(O)C(O)=C1'}

# 同类物质固定相同颜色: SMILES -> 类别 -> 固定颜色索引 (与 spec2vec_tsne.py / msbert.py 一致)
SMILE_CLASS = {
    'O=C(OC1CC2C(O)(CC=C3CC(O)CCC32C)C4(O)CCC(O)(C(=O)C)C14C)C5=CC=C(O)C=C5': 'Steroid',
    'O=C1C=C(OC=2C=C(O)C=C(O)C12)C=3C=CC=CC3': 'Flavonoid',
    'O=C(OC)C1(O)N2C=3C=CC=CC3C4=C2C5N(CC4)CCCC5(CC)C1': 'Alkaloid',
    'OCC1OC(OC2C(O)C(O)C(OC2OC3CCC4(C)C(CCC5(C)C4CC(O)C6C(CCC65C)C(O)(C)CCC=C(C)C)C3(C)C)CO)C(O)C(O)C1O': 'Saponin',
    'O=C(O)C12CCC(C)C(C)C2C3=CCC4C5(C)CC(O)C(O)C(C)(CO)C5CCC4(C)C3(C)CC1': 'Triterpene',
    'O=C1OC=2C(C=C1)=CC=3C=COC3C2OCC(O)C(O)(C)C': 'Coumarin',
    'O=C1C=2C(O)=CC(O)=CC2OC(C3=CC=C4OC(C5=CC=C(OC)C(O)=C5)C(OC4=C3)CO)C1O': 'Flavonoid',
    'O=C(OC1=C(OC(=O)C)C2=C(C=C1C(C)C)CCC3C2(C(=O)OC)CCCC3(C)C)C': 'Diterpene',
    'O=C1CCC2C(C)(CCC3C2(C)CCC4(C)C5CC(C)(C)CCC5(C)CCC34C)C1C': 'Steroid',
    'O=C(O)C=CC1=CC=C(O)C(O)=C1': 'Phenolic acid',
}
CLASS_COLOR = {'Steroid': 0, 'Flavonoid': 1, 'Alkaloid': 2, 'Saponin': 3,
               'Triterpene': 4, 'Coumarin': 5, 'Diterpene': 6, 'Phenolic acid': 7}


if __name__ == '__main__':
    model = MSDiffAug_MaskTop5PlusRandom(100002, 512, 6, 16, 0, 100, 3)
    model.load_state_dict(torch.load(MODEL_FILE))

    ref, test_msms1, test_precursor1, ref_smiles = ParseOrbitrap(DATA_PATH)
    ref_data, _ = MakeTrainData(test_msms1, test_precursor1, MAX_LEN)
    ref_data = np.array(ref_data)

    # # 统计每个SMILES出现的次数
    # smile_counts = Counter(ref_smiles)
    #
    # # 选择出现次数超过100次的SMILES
    # frequent_smiles = [smile for smile, count in smile_counts.items() if count > 100]
    #
    # # 从频繁出现的SMILES中随机选择10个
    # np.random.seed(42)  # 保证结果可重复
    # SELECTED_SMILES = np.random.choice(frequent_smiles, size=10, replace=False)
    #
    # print("选取的10个SMILES:", selected_smiles)

    ref_10 = [i for i, smile in enumerate(ref_smiles) if smile in SELECTED_SMILES]
    colors = [CLASS_COLOR[SMILE_CLASS[ref_smiles[i]]] for i in ref_10]

    msdiff_arr = ModelEmbed(model, ref_data[ref_10], 64)

    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(msdiff_arr)

    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=colors, cmap='tab10', s=2)
    plt.savefig(OUT_PNG, dpi=600, bbox_inches='tight')
    plt.savefig(OUT_PDF, format='pdf', bbox_inches='tight')
    print(f'MSDiffAug: {len(ref_10)} 条谱, 图已保存: {OUT_PNG}')
