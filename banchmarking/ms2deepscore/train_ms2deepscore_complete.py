# -*- coding: utf-8 -*-
"""
用自定义数据集训练 MS2DeepScore 模型的完整脚本。

数据要求（matchms Spectrum 对象列表）：
  - 每张谱图需有 peaks 和 precursor_mz 元数据
  - 每张谱图需有 smiles / inchi / inchikey 之一（用于计算 Tanimoto 训练标签）
  - 若数据同时含正负离子模式谱图，则自动启用跨离子模式训练，此时需有 ionmode 元数据
"""
import os
import pickle
import sys

import numpy as np
import torch
from rdkit import Chem
from tqdm import tqdm

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from banchmarking.ms2deepscore.matchms_compat import (DEFAULT_FILTERS,
                                                    SpectrumProcessor,
                                                    load_spectra)

from  banchmarking.ms2deepscore.SettingsMS2Deepscore import SettingsMS2Deepscore
from models.SiameseSpectralModel import (SiameseSpectralModel,
                                                      train)
from train_new_model.data_generators import create_data_generator
from train_new_model.validation_and_test_split import \
    split_spectra_in_random_inchikey_sets
from  banchmarking.ms2deepscore.utills.ValidationLossCalculator import \
    ValidationLossCalculator

# ==================== 修改成你自己的数据路径 ====================
DATA_PATH = r"/home/ausu/projects/yfs/data/GNPSdata_f2/ob_train_ref.pickle"
OUTPUT_MODEL_FILE = "ms2deepscore_trainedt.model"
# ==============================================================

# 固定随机种子，使数据划分、训练对采样、数据增强可复现
RANDOM_SEED = 55

# 验证集固定抽样上限：验证损失每 epoch 计算 O(N^2) 矩阵，抽样可大幅降低内存与耗时
MAX_VAL_SPECTRA = 3000


def load_spectra_from_path(path):
    """按扩展名自动加载谱图：.mgf/.msp/.mzml 用 matchms，.pickle/.pkl 用 pickle"""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".pickle", ".pkl"):
        with open(path, "rb") as f:
            spectra = pickle.load(f)
        # 兼容 [[specs...], [specs...]] 嵌套列表格式
        if spectra and isinstance(spectra[0], list):
            spectra = [s for sub in spectra for s in sub]
    else:
        spectra = list(tqdm(load_spectra(path), desc="Loading spectra"))
    return [s for s in spectra if s is not None]


def ensure_inchikey(spectrum):
    """谱图缺少 inchikey 时，尝试从 inchi / smiles 推导（训练流程必须要有 inchikey）"""
    if spectrum.get("inchikey"):
        return spectrum
    inchi = spectrum.get("inchi")
    mol = Chem.MolFromInchi(inchi) if inchi else None
    if mol is None and spectrum.get("smiles"):
        mol = Chem.MolFromSmiles(spectrum.get("smiles"))
    if mol is not None:
        spectrum.set("inchikey", Chem.MolToInchiKey(mol))
    return spectrum


def has_both_ionmodes(spectra):
    """判断数据是否同时包含正、负离子模式谱图"""
    modes = {s.get("ionmode") for s in spectra}
    return "positive" in modes and "negative" in modes


def filter_valid_ionmodes(spectra):
    """仅保留 ionmode 为 positive / negative 的谱图（CategoricalToBinary 遇到其他值会报错）"""
    return [s for s in spectra if s.get("ionmode") in ("positive", "negative")]


def main():
    # 1. 加载数据
    spectra = load_spectra_from_path(DATA_PATH)
    print(f"Loaded {len(spectra)} spectra")

    # 2. matchms 默认过滤流程（与官方教程一致）
    spectrum_cleaner = SpectrumProcessor(DEFAULT_FILTERS)
    spectra, processing_report = spectrum_cleaner.process_spectra(spectra)
    print(processing_report)
    print(f"After filtering: {len(spectra)} spectra")

    # 3. 确保每张谱图都有 inchikey
    spectra = [ensure_inchikey(s) for s in spectra]
    spectra = [s for s in spectra if s.get("inchikey")]
    print(f"With inchikey: {len(spectra)} spectra")

    # 4. 按 inchikey 划分训练/验证/测试集（验证集和测试集各约 1/10）
    val_spectra, test_spectra, train_spectra = split_spectra_in_random_inchikey_sets(
        spectra, k=10, random_seed=RANDOM_SEED)
    print(f"train: {len(train_spectra)}, val: {len(val_spectra)}, test: {len(test_spectra)}")

    # 5. 训练设置：前体 m/z 作为数值元数据输入；若含正负离子模式，离子模式也作为元数据输入
    cross_ionmode = has_both_ionmodes(train_spectra)
    if cross_ionmode:
        # 离子模式作为元数据输入时只接受 positive / negative，先过滤掉无效离子模式的谱图
        train_spectra = filter_valid_ionmodes(train_spectra)
        val_spectra = filter_valid_ionmodes(val_spectra)
        test_spectra = filter_valid_ionmodes(test_spectra)
        additional_metadata = [
            ("CategoricalToBinary", {"metadata_field": "ionmode",
                                     "entries_becoming_one": "positive",
                                     "entries_becoming_zero": "negative"}),
            ("StandardScaler", {"metadata_field": "precursor_mz",
                                "mean": 0, "standard_deviation": 1000}),
        ]
    else:
        additional_metadata = [
            ("StandardScaler", {"metadata_field": "precursor_mz",
                                "mean": 0, "standard_deviation": 1000}),
        ]
    # random_seed 会传递给配对选择与 SpectrumPairGenerator（含数据增强），
    # 并在 SettingsMS2Deepscore 内部执行 np.random.seed
    # base_dims=(2000, 2000, 2000) 为官方文档默认结构（首层 9900->2000，约 2900 万参数），
    # 相比代码默认的 (10000,)（约 1 亿参数）显存与计算量均降低约 5 倍
    settings = SettingsMS2Deepscore(additional_metadata=additional_metadata,
                                    random_seed=RANDOM_SEED,
                                    base_dims=(2000, 2000, 2000),
                                    embedding_dim=400,
                                    batch_size=128)

    # 固定 PyTorch 侧的随机性（权重初始化等），需在创建模型之前调用
    # torch.manual_seed(42)
    torch.manual_seed(RANDOM_SEED)

    model = SiameseSpectralModel(settings=settings)

    # 6. 构造训练数据生成器（内部完成 inchikey 配对选择、批生成与数据增强）
    train_generator = create_data_generator(train_spectra, settings)

    # 6.5 验证集固定抽样（种子固定，可复现；验证损失只用于早停与保存最优模型）
    if len(val_spectra) > MAX_VAL_SPECTRA:
        val_rng = np.random.default_rng(RANDOM_SEED)
        val_idx = val_rng.choice(len(val_spectra), MAX_VAL_SPECTRA, replace=False)
        val_spectra = [val_spectra[i] for i in sorted(val_idx)]
        print(f"Validation set subsampled to {len(val_spectra)} spectra")

    # 7. 验证损失计算器
    validation_loss_calculator = ValidationLossCalculator(val_spectra, settings)

    # 8. 训练并保存模型
    history = train(
        model,
        train_generator,
        num_epochs=settings.epochs,
        learning_rate=settings.learning_rate,
        validation_loss_calculator=validation_loss_calculator,
        patience=settings.patience,
        loss_function=settings.loss_function,
        checkpoint_filename=OUTPUT_MODEL_FILE,
        lambda_l1=0,
        lambda_l2=0,
    )
    print("Training finished. Model saved to:", OUTPUT_MODEL_FILE)


if __name__ == "__main__":
    main()
