# -*- coding: utf-8 -*-
"""化学合理性 (Chemical plausibility) 与峰级一致性 (Peak-level) 指标计算。

服务于 ConsistencyAnalysis/ChemicalPlausibilityAnalysis 实验：
- 化学合理性: 原谱 (original 对照) / 扩散增广 (diff) / 随机偏移 (random) 在
  五项指标 (PRC / ISC / PNR / SNL / FNL) 上的对比；ISC 进一步按原谱峰强度
  升序三等分为 Low/Med/High 分层。
- 峰级一致性: 原谱 / 随机增广 / 扩散增广 的保留率 CPR / 新峰率 NPR /
  谱内一致性 ISC / 余弦相似度 / m/z 位移分布。
词表: [PAD]=0, [MASK]=1, m/z token = id-2 对应 0.00~999.99 Da (0.01 Da/格)。
"""
import math
from collections import defaultdict

import numpy as np
from scipy import sparse

VOCAB_SIZE = 100002
BIN_WIDTH = 0.01
MZ_TOL = 0.02  # 匹配容差 (Da)，与词表分辨率 0.01 Da/bin 对应 (2 token)
NL_TOL = MZ_TOL

# C1: 前体质量约束 —— 碎片质量合理范围
MIN_FRAG_MZ = 15.0  # 最小碎片质量 (CH3+ ≈ 15.02 Da)
ADDUCT_TOL = 1.03   # 前体上界余量: 质子加合物 1.0078 + 容差 0.02 Da

# 峰级 ISC 谱内一致性匹配容差 (Da): 新生成峰落在其他原有谱峰 ±ISC_TOL 内
ISC_TOL = 0.2

GROUP_NAMES = ["high", "medium", "low"]

# PNR/SNL/FNL: 常见中性丢失 (Da)
COMMON_NEUTRAL_LOSSES = [
      # ---- 氢 / 无机小分子 ----
      1.01,   # H• 氢自由基
      2.02,   # H2 氢气
      15.02,  # CH3• 甲基自由基(甲基醚、N-甲基、咖啡因类)
      16.03,  # CH4 甲烷
      17.00,  # OH• 羟基自由基(酚、氢过氧化物)
      17.03,  # NH3 氨(伯胺、酰胺、氨基酸侧链)
      18.01,  # H2O 水(醇、羧基脱水;最常见)
      19.00,  # F• 氟自由基(多氟芳烃)
      20.01,  # HF 氟化氢(含氟化合物)
      26.02,  # C2H2 乙炔
      27.01,  # HCN 氰化氢(腈、含氮杂环)
      27.99,  # CO 一氧化碳(醛、酮、黄嘌呤)
      28.01,  # N2 氮气(叠氮、重氮化合物)
      28.03,  # C2H4 乙烯(乙基化合物)
      29.04,  # C2H5• 乙基自由基
      30.00,  # NO 一氧化氮(硝基、硝酸酯)
      30.01,  # CH2O 甲醛(芳甲醚、羟甲基)
      30.05,  # C2H6 乙烷
      31.04,  # CH3NH2 甲胺(甲基酰胺)
      31.99,  # O2 氧气(过氧化物)
      32.03,  # CH3OH 甲醇(甲酯)
      32.98,  # HS• 巯基自由基(硫醇)
      33.00,  # OOH• 过氧羟基自由基(氢过氧化物)
      33.99,  # H2S 硫化氢(硫醇)
      34.97,  # Cl• 氯自由基(多氯化合物)
      35.04,  # NH3+H2O 联合丢失
      35.98,  # HCl 氯化氢(氯代物)
      36.02,  # 2H2O 双脱水(邻二醇、糖)
      40.03,  # C3H4 丙炔/丙二烯(炔类)
      41.03,  # CH3CN 乙腈
      41.04,  # C3H5• 烯丙基自由基
      42.01,  # CH2CO 乙烯酮(酚乙酸酯、N-乙酰基)
      42.05,  # C3H6 丙烯(异丙基)
      43.01,  # HNCO 异氰酸(尿素类)
      43.05,  # C3H7• 丙基自由基
      43.99,  # CO2 二氧化碳(羧酸脱羧;最常见)
      44.03,  # CH3CHO 乙醛(乙氧基)
      45.02,  # HCONH2 甲酰胺
      45.06,  # HN(CH3)2 二甲胺 / C2H5NH2 乙胺(酰胺类)
      45.99,  # NO2 二氧化氮(硝基化合物)
      46.01,  # HCOOH 甲酸(甲酸酯;或 CO+H2O)
      46.04,  # C2H5OH 乙醇(乙酯)
      47.00,  # HNO2 亚硝酸 / CH3S• 甲硫基自由基
      47.97,  # SO 一氧化硫(亚砜)
      48.00,  # CH3SH 甲硫醇(硫醚)
      49.99,  # CH3Cl 氯甲烷(氯甲基醚)
      53.03,  # C3H3N 丙烯腈(腈类)
      54.03,  # 3H2O 三次脱水(多羟基、甾体)
      54.05,  # C4H6 丁二烯(逆 Diels-Alder)
      55.05,  # C4H7• 丁烯基自由基
      56.06,  # C4H8 丁烯(叔丁基)
      57.02,  # C2H3NO 甲基异氰酸酯(咖啡因、黄嘌呤)
      57.07,  # C4H9• 丁基自由基(叔丁基)
      58.04,  # C3H6O 丙酮
      59.04,  # CH3CONH2 乙酰胺(N-乙酰基)
      59.07,  # N(CH3)3 三甲胺 / C3H9N 丙胺(季铵、磷脂酰胆碱)
      60.02,  # CH3COOH 乙酸(乙酰氧基)
      60.03,  # CH4N2O 尿素(脲类)
      60.06,  # C3H8O 丙醇/异丙醇(丙酯)
      61.99,  # NO3• 硝酸根自由基(硝酸酯)
      62.00,  # H2CO3 碳酸(碳酸酯)
      62.02,  # C2H6S 乙硫醇(硫醚)
      62.04,  # C2H6O2 乙二醇(PEG、酯)
      63.00,  # HNO3 硝酸(硝酸酯)
      63.96,  # SO2 二氧化硫(磺酰胺、磺酸酯)
      66.05,  # C5H6 环戊二烯
      68.06,  # C5H8 异戊二烯/环戊烯
      69.00,  # CF3• 三氟甲基自由基(三氟甲基化合物)
      69.07,  # C5H9• 戊烯基自由基
      70.08,  # C5H10 戊烯(戊基)
      71.04,  # C3H5NO 乙基异氰酸酯(可可碱、黄嘌呤)
      71.09,  # C5H11• 戊基自由基
      72.02,  # C3H4O2 丙烯酸(丙烯酸酯)
      72.06,  # C4H8O 丁醛(丁氧基)
      74.04,  # C3H6O2 丙酸(丙酸酯)
      74.07,  # C4H10O 丁醇(丁酯)
      75.03,  # C2H5NO2 甘氨酸(甘氨酸结合物)
      76.05,  # C3H8O2 丙二醇(丙二醇酯)
      77.04,  # C6H5• 苯基自由基
      78.05,  # C6H6 苯
      78.92,  # Br• 溴自由基
      78.96,  # PO3• 磷酸根自由基(磷酸酯)
      79.04,  # C5H5N 吡啶(吡啶类)
      79.93,  # HBr 溴化氢(溴代物)
      79.96,  # SO3 三氧化硫(硫酸酯结合物)
      79.97,  # HPO3 偏磷酸(磷酸化肽、核苷酸)
      82.08,  # C6H10 环己烯(逆 Diels-Alder)
      84.09,  # C6H12 己烯(己基)
      85.10,  # C6H13• 己基自由基
      86.07,  # C5H10O 戊醛(戊氧基)
      88.05,  # C4H8O2 丁酸(丁酸酯)
      88.09,  # C5H12O 戊醇(戊酯)
      89.05,  # C3H7NO2 丙氨酸(氨基酸结合物)
      90.03,  # C3H6O3 甘油醛片段(糖 cross-ring 断裂)
      91.05,  # C7H7• 苄基自由基
      92.06,  # C7H8 甲苯
      93.06,  # C6H7N 苯胺(苯胺类)
      94.04,  # C6H6O 苯酚(酚醚/酚酯)
      97.97,  # H2SO4 硫酸
      97.98,  # H3PO4 磷酸(pSer/pThr、磷酸酯)
      99.12,  # C7H15• 庚基自由基
      102.07, # C5H10O2 戊酸(戊酸酯)
      102.10, # C6H14O 己醇(己酯)
      104.06, # C8H8 苯乙烯
      105.03, # C6H5CO• 苯甲酰自由基(苯甲酸酯)
      105.04, # C3H7NO3 丝氨酸(丝氨酸结合物)
      106.04, # C7H6O 苯甲醛(苄醚)
      108.06, # C7H8O 苯甲醚/甲酚(芳甲醚)
      110.04, # C6H6O2 邻苯二酚(儿茶酚醚)
      111.04, # 胞嘧啶 C4H5N3O(核酸碱基)
      112.03, # 尿嘧啶 C4H4N2O2(核酸碱基)
      113.13, # C8H17• 辛基自由基
      113.99, # CF3COOH 三氟乙酸(TFA 衍生物)
      116.08, # C6H12O2 己酸(己酸酯)
      117.06, # C8H7N 吲哚(吲哚类)
      120.04, # C4H8O4 四碳糖片段(糖 cross-ring)
      121.02, # C3H7NO2S 半胱氨酸(半胱氨酸结合物)
      122.04, # C7H6O2 苯甲酸(苯甲酸酯)
      124.05, # C7H8O2 愈创木酚(木质素类)
      125.01, # C2H7NO3S 牛磺酸(牛磺酸结合物)
      126.04, # 胸腺嘧啶 C5H6N2O2(核酸碱基)
      126.90, # I• 碘自由基
      127.91, # HI 碘化氢(碘代物)
      129.04, # C5H7NO3 焦谷氨酸(GSH 结合物)
      130.10, # C7H14O2 庚酸(庚酸酯)
      132.04, # 戊糖 C5H8O4(糖苷)
      134.06, # 脱氧核糖 C5H10O4(核苷)
      135.05, # 腺嘌呤 C5H5N5(核酸碱基)
      136.05, # C8H8O2 苯乙酸(苯乙酸酯)
      138.03, # C7H6O3 水杨酸(水杨酸酯)
      141.02, # 磷酸乙醇胺 C2H8NO4P(磷脂酰乙醇胺)
      144.12, # C8H16O2 辛酸(辛酸酯)
      146.06, # 岩藻糖 C6H10O4(糖链)
      147.05, # C5H9NO4 谷氨酸(GSH/氨基酸结合物)
      150.05, # C5H10O5 核糖 / 糖 cross-ring(核苷、糖链)
      151.05, # 鸟嘌呤 C5H5N5O(核酸碱基)
      152.05, # C8H8O3 香草醛(香草醛类)
      162.05, # 己糖/肌醇 C6H10O5(糖链、糖苷)
      164.05, # C9H8O3 香豆酸(香豆酸酯)
      176.03, # 葡萄糖醛酸 C6H8O6(II 相代谢物)
      178.04, # C5H10N2O3S 半胱氨酰甘氨酸(GSH 代谢物)
      180.06, # C6H12O6 葡萄糖(双糖苷)
      184.07, # 磷酸胆碱 C5H15NO4P(磷脂酰胆碱)
      185.01, # 磷酸丝氨酸 C3H8NO6P(磷脂酰丝氨酸)
      203.08, # HexNAc C8H13NO5(N-糖链)
      228.21, # 肉豆蔻酸 C14H28O2(FA 14:0)
      254.22, # 棕榈油酸 C16H30O2(FA 16:1)
      256.24, # 棕榈酸 C16H32O2(FA 16:0)
      280.24, # 亚油酸 C18H32O2(FA 18:2)
      282.26, # 油酸 C18H34O2(FA 18:1)
      284.27, # 硬脂酸 C18H36O2(FA 18:0)
      291.10, # 唾液酸 NeuAc C11H17NO8(糖链末端)
      304.24, # 花生四烯酸 C20H32O2(FA 20:4)
      307.08, # 谷胱甘肽 C10H17N3O6S(GSH 结合物)
      308.11, # 芸香糖 C12H20O9(芦丁等二糖苷)
      324.11, # 双己糖 C12H20O10(二糖苷)
      328.24, # DHA C22H32O2(FA 22:6)
  ]

NL_ARR = np.array(COMMON_NEUTRAL_LOSSES, dtype=np.float64)


def token_to_mz(tok):
    """token id -> m/z (Da)。"""
    return (int(tok) - 2) * BIN_WIDTH


def match_neutral_loss(value_da, losses=COMMON_NEUTRAL_LOSSES, tol=MZ_TOL):
    """|value_da - NL_j| <= tol 是否成立; 返回匹配的 NL 或 None。"""
    for nl in losses:
        if abs(value_da - nl) <= tol:
            return nl
    return None


def is_common_neutral_loss(delta_da, losses=COMMON_NEUTRAL_LOSSES, tol=NL_TOL):
    """位移量 (Da, 取绝对值) 是否为常见中性丢失。"""
    abs_d = abs(delta_da)
    return any(abs(abs_d - nl) <= tol for nl in losses)


def group_positions(ids, intensity):
    """各强度分组的峰位置 (排除前体位置 0 与 padding)。

    原谱所有碎片峰按强度升序排序并 3 等分 (等数量) 为 low / medium / high
    (各占 1/3), 重构峰按其原谱强度归入相应等级。
    与 np.array_split(order, 3) 分组结果一致。
    """
    peaks = [p for p in range(1, len(ids)) if int(ids[p]) != 0]
    order = sorted(peaks, key=lambda p: float(intensity[p]))
    groups = {"high": [], "medium": [], "low": []}
    n = len(order)
    for i, p in enumerate(order):
        k = min(i * 3 // n, 2)
        groups[("low", "medium", "high")[k]].append(p)
    return groups


def _matched_nls(values):
    """每个数值匹配到的第一个常规中性丢失 (Da)。"""
    out = []
    for v in values:
        for nl in COMMON_NEUTRAL_LOSSES:
            if abs(float(v) - nl) <= MZ_TOL:
                out.append(nl)
                break
    return out


def chemical_plausibility(orig_ids, view_ids, masked_positions, intensity=None):
    """单条谱: 原谱 S vs 增强视图 V 的五项化学合理性指标检查。

    生成峰 (n_gen) = masked 位置中视图非零的峰 (含未改峰);
    新生成峰 (n_N = n_chg) = 生成峰中视图与原值不同的峰 (被改且未消失);
    峰消失 (视图 0) 不计入任何指标。
    原谱对照: view=orig 且 masked=全部真实碎片峰位置 (此时 n_N=0)。

    PRC 前体质量约束: MIN_FRAG_MZ <= m' <= precursor + ADDUCT_TOL, 分母 n_N
    ISC 谱内一致性: m' 落在 S 碎片集合 (位置>=1 的非零峰) 的 ±MZ_TOL 内, 分母 n_N
    PNR 前体中性丢失: m' ≈ precursor - NL_j (±MZ_TOL), 分母 n_gen
    SNL 位移即中性丢失: |m - m'| ≈ NL_j (±MZ_TOL), 分母 n_N
    FNL 碎片中性丢失: m' ≈ s - NL_j (s ∈ S), 落在 S' = {s - NL_j} 的 ±MZ_TOL 内, 分母 n_gen

    ISC 分层 (intensity 提供时): 原谱真实峰按强度升序三等分 (等数量,
    见 group_positions) 为 Low/Med/High; 对每层统计原峰属于该层的新生成峰
    的 ISC 命中率, 分母 n_chg_{low,med,high} (原谱无峰位置新出现的峰不参与分层)。

    返回 dict:
      n_gen / n_chg / c1_ok / c1_low / c1_high / c2_ok / c3_ok / c3_nl /
      c4_ok / c4_nl / c7_ok / n_chg_low / n_chg_med / n_chg_high /
      c2_ok_low / c2_ok_med / c2_ok_high / violations / precursor_mz
    """
    orig = np.asarray([int(x) for x in orig_ids], dtype=np.int64)
    view = np.asarray([int(x) for x in view_ids], dtype=np.int64)
    precursor_mz = token_to_mz(orig[0])

    frag = orig[1:]
    frag_mz = np.unique((frag[frag != 0] - 2) * BIN_WIDTH)

    masked = np.asarray([int(p) for p in masked_positions], dtype=np.int64)
    gen = masked[view[masked] != 0] if masked.size else np.array([], dtype=np.int64)

    n_gen = int(gen.size)
    empty = {
        "n_gen": 0, "n_chg": 0,
        "c1_ok": 0, "c1_low": 0, "c1_high": 0,
        "c2_ok": 0, "c3_ok": 0, "c3_nl": [],
        "c4_ok": 0, "c4_nl": [], "c7_ok": 0,
        "n_chg_low": 0, "n_chg_med": 0, "n_chg_high": 0,
        "c2_ok_low": 0, "c2_ok_med": 0, "c2_ok_high": 0,
        "violations": [], "precursor_mz": precursor_mz,
    }
    if n_gen == 0:
        return empty

    m = (orig[gen] - 2) * BIN_WIDTH    # 原质量
    mp = (view[gen] - 2) * BIN_WIDTH   # 生成质量

    # 新生成峰 (被改且未消失): PRC/ISC/SNL 的分母 n_N
    chg_mask = view[gen] != orig[gen]
    n_chg = int(np.sum(chg_mask))

    # PRC: 前体质量约束 (新生成峰)
    low_mask = mp < MIN_FRAG_MZ
    high_mask = mp > precursor_mz + ADDUCT_TOL
    c1_ok = int(np.sum(chg_mask & ~(low_mask | high_mask)))
    c1_low = int(np.sum(chg_mask & low_mask))
    c1_high = int(np.sum(chg_mask & high_mask))
    violations = [(float(a), float(b), "too_low")
                  for a, b in zip(m[chg_mask & low_mask], mp[chg_mask & low_mask])]
    violations += [(float(a), float(b), "too_high")
                   for a, b in zip(m[chg_mask & high_mask], mp[chg_mask & high_mask])]

    # ISC: 谱内一致性 (新生成峰)
    c2_mask = (np.abs(mp[:, None] - frag_mz[None, :]) <= MZ_TOL).any(axis=1)
    c2_ok = int(np.sum(chg_mask & c2_mask))

    # ISC 分层: 原谱真实峰按强度升序三等分 (等数量) 为 Low/Med/High,
    # 分母 = 原峰属于该层的新生成峰数, 分子 = 其中落在 S ±MZ_TOL 内的数量
    n_chg_low = n_chg_med = n_chg_high = 0
    c2_ok_low = c2_ok_med = c2_ok_high = 0
    if intensity is not None:
        inten = np.asarray([float(x) for x in intensity], dtype=np.float64)
        groups = group_positions(orig, inten)
        lvl = np.full(orig.shape, -1, dtype=np.int64)  # -1: 原谱无峰位置, 不参与分层
        for g, positions in groups.items():
            lvl[positions] = ("low", "medium", "high").index(g)
        lvl_chg = lvl[gen][chg_mask]
        c2m = c2_mask[chg_mask]
        n_chg_low = int(np.sum(lvl_chg == 0))
        n_chg_med = int(np.sum(lvl_chg == 1))
        n_chg_high = int(np.sum(lvl_chg == 2))
        c2_ok_low = int(np.sum(c2m & (lvl_chg == 0)))
        c2_ok_med = int(np.sum(c2m & (lvl_chg == 1)))
        c2_ok_high = int(np.sum(c2m & (lvl_chg == 2)))

    # PNR: 前体中性丢失 (生成峰)
    c3_mask = (np.abs(mp[:, None] - (precursor_mz - NL_ARR)[None, :]) <= MZ_TOL).any(axis=1)
    c3_ok = int(np.sum(c3_mask))
    c3_nl = _matched_nls(precursor_mz - mp[c3_mask])

    # SNL: 位移即中性丢失 (新生成峰)
    if n_chg:
        disp = np.abs(mp[chg_mask] - m[chg_mask])
        c4_mask = (np.abs(disp[:, None] - NL_ARR[None, :]) <= MZ_TOL).any(axis=1)
        c4_ok = int(np.sum(c4_mask))
        c4_nl = _matched_nls(disp[c4_mask])
    else:
        c4_ok, c4_nl = 0, []

    # FNL: 碎片中性丢失 (生成峰) — m' 落在 S' = {s - NL_j : s ∈ S} 的 ±MZ_TOL 内
    frag_shifted = np.unique((frag_mz[None, :] - NL_ARR[:, None]).ravel())
    c7_mask = (np.abs(mp[:, None] - frag_shifted[None, :]) <= MZ_TOL).any(axis=1)
    c7_ok = int(np.sum(c7_mask))

    return {
        "n_gen": n_gen, "n_chg": n_chg,
        "c1_ok": c1_ok, "c1_low": c1_low, "c1_high": c1_high,
        "c2_ok": c2_ok,
        "c3_ok": c3_ok, "c3_nl": c3_nl,
        "c4_ok": c4_ok, "c4_nl": c4_nl,
        "c7_ok": c7_ok,
        "n_chg_low": n_chg_low, "n_chg_med": n_chg_med, "n_chg_high": n_chg_high,
        "c2_ok_low": c2_ok_low, "c2_ok_med": c2_ok_med, "c2_ok_high": c2_ok_high,
        "violations": violations,
        "precursor_mz": precursor_mz,
    }


def ids_to_vec(ids, intensity, start=1):
    """一条谱 -> token 空间稀疏向量 dict {token: 强度}，默认跳过前体位置 0。"""
    vec = defaultdict(float)
    for p in range(start, len(ids)):
        tok = int(ids[p])
        if tok == 0:
            continue
        vec[tok] += float(intensity[p])
    return dict(vec)


def ids_to_vec_at(ids, intensity, positions):
    """只在指定位置 (如 masked 位置) 上构建 token 空间稀疏向量。"""
    vec = defaultdict(float)
    for p in positions:
        tok = int(ids[p])
        if tok == 0:
            continue
        vec[tok] += float(intensity[p])
    return dict(vec)


def vec_cosine(v1, v2):
    """稀疏向量余弦相似度。"""
    if not v1 or not v2:
        return 0.0
    common = v1.keys() & v2.keys()
    dot = sum(v1[t] * v2[t] for t in common)
    n1 = math.sqrt(sum(x * x for x in v1.values()))
    n2 = math.sqrt(sum(x * x for x in v2.values()))
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    return dot / (n1 * n2)


def build_sparse_matrix(vecs, dim=VOCAB_SIZE):
    """dict 向量列表 -> L2 行归一化的 csr_matrix (行=谱, 列=token)。"""
    rows, cols, data = [], [], []
    for i, v in enumerate(vecs):
        for tok, w in v.items():
            rows.append(i)
            cols.append(tok)
            data.append(w)
    mat = sparse.csr_matrix((data, (rows, cols)), shape=(len(vecs), dim), dtype=np.float32)
    norms = np.sqrt(np.asarray(mat.power(2).sum(axis=1)).ravel())
    norms = np.where(norms == 0, 1.0, norms)
    mat.data /= np.repeat(norms, np.diff(mat.indptr))
    return mat


def spectrum_metrics(orig_ids, intensity, view_ids, masked_positions, bin_mode=False):
    """单条谱: 原谱 vs 一个增强视图 的峰级指标。

    论文口径: 锚点峰 = 未被 mask 的位置 (m/z 固定);
    重构峰 n_R = 被 mask 的非零峰; 保留峰 n_C = 修改后 m/z 与原谱一致 (精确匹配);
    新生成峰 n_N = 修改后 m/z 发生变化的峰 (含峰消失与变为原谱已有 token);
    恒有 n_R = n_C + n_N, 故 CPR + NPR = 1。

    bin_mode: 低分辨率基线 (VAE, 0.1 Da bin) 的对齐口径, 无容差窗口——
    保留 = view 在原峰所在 bin 有输出峰 (view 非零), 被改 = 同 bin 无峰 (峰消失,
    view 为 0, 故无位移可统计, ISC 恒为 0)。

    conserved: 按原谱强度 3 等分组 (high/medium/low) 返回各组 (保留数, 重构数),
    由上层全局汇总 CPR_l = Σn_C,l / Σn_R,l (锚点峰不参与统计)。
    conserved_tol: 同上但 |m' - m| <= MZ_TOL (0.02 Da) 即视为保留 (CPR.02),
    供低分辨率基线 (如 0.1 Da bin 的 VAE) 公平比较; bin_mode 下与 conserved
    同口径 (bin 即其原生分辨率, 不再叠加容差)。
    masked_changed_counts / new_peak_counts: (被改数, 重构总数) 计数, 两者相同,
    供上层算 CPR = 1 - Σ被改/Σn_R 与 NPR = Σ被改/Σn_R。
    masked_cosine: 只在 masked 位置上的强度加权余弦。
    isc_counts: (命中数, n_N) —— ISC 谱内一致性: 新生成峰 (含消失) 中,
    视图 m/z 落在其他原有谱峰 ±ISC_TOL 内的比例 (排除自身原位置匹配,
    消失峰计为未命中)。
    deviations: 被扰动且视图仍为有效峰的 m/z 位移 (Da, 带符号);
    视图该位置为 0 (峰消失) 的计入 n_N 但不算位移。
    """
    groups = group_positions(orig_ids, intensity)

    masked = [int(p) for p in masked_positions]
    masked_set = set(masked)

    conserved = {}
    conserved_tol = {}
    for gname, positions in groups.items():
        pos = [p for p in positions if p in masked_set]
        if bin_mode:
            conserved[gname] = (float(sum(1 for p in pos if int(view_ids[p]) != 0)),
                                float(len(pos)))
            conserved_tol[gname] = conserved[gname]
        else:
            conserved[gname] = (float(sum(1 for p in pos if int(view_ids[p]) == int(orig_ids[p]))),
                                float(len(pos)))
            conserved_tol[gname] = (float(sum(1 for p in pos if abs(int(view_ids[p]) - int(orig_ids[p])) <= MZ_TOL / BIN_WIDTH)),
                                    float(len(pos)))

    if masked_set:
        masked_cosine = vec_cosine(ids_to_vec_at(orig_ids, intensity, masked_set),
                                   ids_to_vec_at(view_ids, intensity, masked_set))
    else:
        masked_cosine = float("nan")

    if bin_mode:
        masked_changed = [p for p in masked if int(view_ids[p]) == 0]
    else:
        masked_changed = [p for p in masked if int(view_ids[p]) != int(orig_ids[p])]
    n_masked = len(masked)
    new_peaks = masked_changed
    new_peak_ratio = len(new_peaks) / n_masked if n_masked else 0.0

    # ISC: 新生成峰 (含消失) 中, 视图 m/z 落在其他原有谱峰 ±ISC_TOL 内的比例
    peaks = [p for p in range(1, len(orig_ids)) if int(orig_ids[p]) != 0]
    peaks_mz = np.asarray([(int(orig_ids[p]) - 2) * BIN_WIDTH for p in peaks], dtype=np.float64)
    peaks_pos = np.asarray(peaks)
    n_isc_hit = 0
    for p in masked_changed:
        if int(view_ids[p]) == 0:
            continue  # 峰消失: 计入分母, 计为未命中
        mp = (int(view_ids[p]) - 2) * BIN_WIDTH
        close = np.abs(peaks_mz - mp) <= ISC_TOL
        close &= peaks_pos != p  # 排除自身原位置
        if close.any():
            n_isc_hit += 1

    deviations = [(int(view_ids[p]) - int(orig_ids[p])) * BIN_WIDTH
                  for p in masked_changed if int(view_ids[p]) != 0]

    cos = vec_cosine(ids_to_vec(orig_ids, intensity), ids_to_vec(view_ids, intensity))

    return {
        "conserved": conserved,
        "conserved_tol": conserved_tol,
        "masked_changed_counts": (float(len(masked_changed)), float(n_masked)),
        "masked_cosine": masked_cosine,
        "new_peak_counts": (float(len(new_peaks)), float(n_masked)),
        "new_peak_ratio": new_peak_ratio,
        "isc_counts": (float(n_isc_hit), float(len(masked_changed))),
        "deviations": deviations,
        "weighted_cosine": cos,
        "masked_changed": masked_changed,
    }
