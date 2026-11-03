# -*- coding: utf-8 -*-
"""
matchms 0.18 兼容层。

本文件夹内代码按新版 matchms (>=0.24) 编写，本模块在 matchms 0.18.0 上提供等价接口：
- load_spectra / save_spectra：按扩展名分发到 matchms 0.18 的 load_from_* / save_as_*
- SpectrumProcessor + DEFAULT_FILTERS：用 0.18 存在的过滤函数拼出官方默认流程
- derive_fingerprint_from_smiles：优先用 matchms 自带的私有函数，缺失时用 RDKit 本地实现
"""
import os
import pickle
from typing import List

import numpy as np

from matchms import Spectrum
from matchms.exporting import save_as_mgf, save_as_msp
from matchms.importing import load_from_json, load_from_mgf, load_from_msp

try:
    from matchms.importing import load_from_mzml
except ImportError:  # 0.18 加载 .mzml 依赖 pyteomics，缺失时禁用
    load_from_mzml = None

try:
    from matchms.filtering import _derive_fingerprint_from_smiles as _matchms_derive_fingerprint
except ImportError:
    _matchms_derive_fingerprint = None


def load_spectra(path, metadata_harmonization=True):
    """按扩展名加载谱图，返回 Spectrum 列表（含 .pickle/.pkl 支持）。"""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".pickle", ".pkl"):
        with open(path, "rb") as f:
            spectra = pickle.load(f)
        if spectra and isinstance(spectra[0], list):
            spectra = [s for sub in spectra for s in sub]
        return [s for s in spectra if s is not None]
    if ext == ".mgf":
        return list(load_from_mgf(path, metadata_harmonization=metadata_harmonization))
    if ext == ".msp":
        return list(load_from_msp(path, metadata_harmonization=metadata_harmonization))
    if ext == ".json":
        return list(load_from_json(path, metadata_harmonization=metadata_harmonization))
    if ext == ".mzml":
        if load_from_mzml is None:
            raise ImportError("matchms 0.18 加载 .mzml 需要 pyteomics，当前环境未安装")
        return list(load_from_mzml(path, metadata_harmonization=metadata_harmonization))
    raise ValueError(f"不支持的谱图文件扩展名: {ext}")


def save_spectra(spectra, path):
    """按扩展名保存谱图。"""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".mgf":
        save_as_mgf(spectra, path)
    elif ext == ".msp":
        save_as_msp(spectra, path)
    else:
        raise ValueError(f"不支持的谱图文件扩展名: {ext}")


def _build_default_filters():
    """用 matchms 0.18 存在的过滤函数拼出官方默认流程（与新版 DEFAULT_FILTERS 等价）。"""
    import matchms.filtering as mf

    filter_names = [
        "make_charge_int",
        "make_ionmode",
        "set_ionmode_na_when_missing",
        "correct_charge",
        "harmonize_undefined_inchikey",
        "harmonize_undefined_inchi",
        "harmonize_undefined_smiles",
        "add_precursor_mz",
        "add_retention_time",
        "normalize_intensities",
        "select_by_mz",
        "select_by_relative_intensity",
        "reduce_to_number_of_peaks",
        "require_minimum_number_of_peaks",
    ]
    filters = []
    for name in filter_names:
        func = getattr(mf, name, None)
        if func is not None:
            filters.append(func)
        else:
            print(f"[matchms_compat] matchms 0.18 缺少过滤函数 {name}，已跳过")
    return filters


DEFAULT_FILTERS = _build_default_filters()


class SpectrumProcessor:
    """matchms>=0.19 的 SpectrumProcessor 的本地等价实现。

    逐个谱图、逐个过滤器执行；过滤抛异常或返回 None 的谱图被剔除，
    processing_report 记录每步过滤前后的谱图数。
    """

    def __init__(self, filters: List):
        self.filters = filters

    def process_spectra(self, spectra: List[Spectrum]):
        report_lines = []
        current = list(spectra)
        for i, filter_func in enumerate(self.filters):
            name = getattr(filter_func, "__name__", f"filter_{i}")
            kept = []
            failed = 0
            for spectrum in current:
                try:
                    result = filter_func(spectrum)
                except Exception:
                    result = None
                if result is None:
                    failed += 1
                else:
                    kept.append(result)
            report_lines.append(f"{name}: {len(current)} -> {len(kept)} (removed {failed})")
            current = kept
        report = "\n".join(report_lines)
        return current, report


def derive_fingerprint_from_smiles(smiles, fingerprint_type="daylight", nbits=2048):
    """从 SMILES 生成指纹 numpy 数组，与 matchms 的 _derive_fingerprint_from_smiles 等价。"""
    if _matchms_derive_fingerprint is not None:
        return _matchms_derive_fingerprint(smiles,
                                           fingerprint_type=fingerprint_type,
                                           nbits=nbits)

    from rdkit import Chem
    from rdkit.Chem import DataStructs

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if fingerprint_type in ("daylight", "rdkit"):
        try:
            from rdkit.Chem import rdFingerprintGenerator
            fp = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=nbits).GetFingerprint(mol)
        except ImportError:
            fp = Chem.RDKFingerprint(mol, fpSize=nbits)
    elif fingerprint_type == "morgan":
        try:
            from rdkit.Chem import rdFingerprintGenerator
            fp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=nbits).GetFingerprint(mol)
        except ImportError:
            from rdkit.Chem import AllChem
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=nbits)
    else:
        raise ValueError(f"不支持的指纹类型: {fingerprint_type}")
    # 旧版 RDKit 的 ConvertToNumpyArray 必须显式传入 destArray，新版可省略；
    # 显式传入在两种版本上都可用
    fp_array = np.zeros((1,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, fp_array)
    return fp_array
