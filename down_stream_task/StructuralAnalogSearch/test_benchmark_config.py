# -*- coding: utf-8 -*-
"""
统一对比测试配置文件 (谱库搜索 + 类似物搜索)

配合 test_benchmark.py 使用, 运行方式:
    python test_benchmark.py

只需改最上面"选择区"里的数字, 不需要手写任何路径。
所有路径在下方"预设表"里已经列好, 要加新的数据或权重就照格式往表里加。
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))

# ================================================================
# 选择区: 只改这里的等号右边
# ================================================================

# ---- 1. 模型 ----
# 1 = MSDiffAug (扩散增广)
# 2 = MSBERT (掩码语言模型)
# 3 = MS2DeepScore (孪生网络)
# 4 = Spec2Vec (词向量)
# 5 = MS/MS 分箱 (binned, 无需训练权重)
# 6 = MSDiffAug MaskTop5PlusRandom (top5内掩1个 + 全有效位置随机掩1个)
MODEL_CHOICE = 6

# ---- 2. 任务 ----
# 1 = 只做谱库搜索   (TOP1/5/10 命中率 + ATS1/5/10)
# 2 = 只做类似物搜索 (全部查询 + 指定 recall 下的 ATS1/5/10)
# 3 = 两个都做       (对下面 DATA_CHOICE 选的同一组数据先跑谱库搜索、再跑类似物搜索,
#                     各输出一套结果; 注意: 谱库搜索配 DATA_CHOICE 1/3 才有意义,
#                     类似物搜索配 2/4/5/6 才有意义, 配错的那个任务数字无意义,
#                     所以一般不选 3, 而是换 DATA_CHOICE 各跑一次任务 1 和任务 2)
TASK_CHOICE = 1

# ---- 3. 数据 ----
# 1 = GNPS 谱库搜索      ob_train_ref.pickle  / ob_train_query.pickle
# 2 = GNPS 类似物搜索     galib.pickle         / gaq0.5_0.95.pickle
# 3 = NIST 谱库搜索       train_ref.pickle     / train_query.pickle
# 4 = NIST 类似物搜索     nalib.pickle         / naq0.5_0.95.pickle
# 5 = NIST->GNPS 交叉    galib.pickle         / NcorossG0.5_0.95.pickle
# 6 = GNPS->NIST 交叉    nalib.pickle         / GcorossN0.5_0.95.pickle

# 7 = GNPS 类似物搜索      ob_train_ref.pickle          / unique_gnps_sim_0.5_0.95.pickle
# 8 = NIST->GNPS 交叉     ob_train_ref.pickle          / unique_nist_sim_0.5_0.95.pickle
# 9 = NIST 类似物搜索      train_ref.pickle        / N2N0.5_0.95.pickle
# 10 = GNPS->NIST 交叉    train_ref.pickle         / G2N0.5_0.95.pickle
DATA_CHOICE = 1

# ---- 4. 谱库嵌入方式 ----
# 1 = 谱库从头嵌入并保存 npz (from-scratch, 覆盖旧 npz)
# 2 = 谱库从已有 npz 加载 (load, 最快)
# 查询嵌入: 每次运行都现场嵌入, 不保存 npz
EMBED_MODE_CHOICE = 1

# ---- 5. 谱库嵌入文件来源 (EMBED_MODE_CHOICE = 2 时生效) ----
# 0 = 用本脚本自动保存的 npz (benchmark_npz 文件夹, 按模型+数据集自动命名)
# 1 = 用各模型旧脚本的 npz (library/N_library* 里的, 按模型+数据自动查下表, 无需写路径)
NPZ_SOURCE_CHOICE = 0

# ---- 6. 模型权重 (选第几个, 从 1 开始; MS/MS 分箱忽略此项) ----选择第几次随机种子
# 权重列表在下方 MODEL_FILES 表里。用 GNPS 组还是 NIST 组由 DATA_CHOICE 自动决定
# (库是 GNPS 就用 GNPS 组权重, 库是 NIST 就用 NIST 组权重)。
MODEL_FILE_CHOICE = 1

# ---- 7. 检索与嵌入参数 ----
DEVICE_CHOICE = 1        # 1 = cuda  2 = cpu
HIT_MODE_CHOICE = 1      # 谱库搜索命中判定: 1 = InChIKey 前14位  2 = SMILES 完全相等
RECALLS = [0.1, 0.3, 0.5, 0.7, 0.9]   # 类似物搜索的覆盖率档位 (0~1, 可增删)
MAX_QUERIES = 0          # 只跑查询集前 N 条 (0 = 全部, 只对查询集生效)
TOP_K = 10               # 每个查询保留前 K 个候选 (TOP 与 ATS 都取 1/5/10 三档)
RETRIEVAL_BATCH = 500    # 检索时查询批大小 (GPU 内存不够调小)
RETRIEVAL_CHUNK = 5000   # 检索时谱库分块大小 (GPU 内存不够调小)
FP_RADIUS = 2            # Morgan 指纹参数 (计算真实结构相似度)
FP_BITS = 2048
MAX_LEN = 100            # MSBERT / MSDiffAug 输入序列长度
EMBED_BATCH = 64         # MSBERT / MSDiffAug 嵌入批大小
SPEC2VEC_N_DECIMALS = 2          # Spec2Vec 专属参数 (与旧脚本一致)
SPEC2VEC_INTENSITY_POWER = 0.5
SPEC2VEC_ALLOWED_MISSING = 5.0
MS2DS_EMBED_BATCH = 32           # MS2DeepScore 嵌入批大小
OUT_DIR = os.path.join(_HERE, "benchmark_out")
OUT_TAG = ""            # 输出 CSV 文件名前缀 (留空 = 用模型名, 如 msdiffaug_analog_per_query.csv;
                        # 填 "xx" 则保存为 xx_analog_per_query.csv)

# ================================================================
# 预设表 (一般不用动; 要加新数据/权重就照格式往表里加一行)
# ================================================================

DATA_COMBOS = {
    1: {"lib": "/home/ausu/projects/yfs/data/GNPSdata_f2/ob_train_ref.pickle",
        "query": "/home/ausu/projects/yfs/data/GNPSdata_f2/ob_train_query.pickle",
        "group": "GNPS", "tag": "gnps_libsearch"},
    2: {"lib": "/home/ausu/projects/yfs/data/GNPSdata_f2/ob_train_ref.pickle",
        "query": "/home/ausu/projects/yfs/data/GNPSdata_f2/gaq0.5_0.95.pickle",
        "group": "GNPS", "tag": "gnps_analog"},
    3: {"lib": "/home/ausu/projects/yfs/data/NistData/train_ref.pickle",
        "query": "/home/ausu/projects/yfs/data/NistData/train_query.pickle",
        "group": "NIST", "tag": "nist_libsearch"},
    4: {"lib": "/home/ausu/projects/yfs/data/NistData/nalib.pickle",
        "query": "/home/ausu/projects/yfs/data/NistData/naq0.5_0.95.pickle",
        "group": "NIST", "tag": "nist_analog"},
    5: {"lib": "/home/ausu/projects/yfs/data/GNPSdata_f2/galib.pickle",
        "query": "/home/ausu/projects/yfs/data/NG_CROSS/NcorossG0.5_0.95.pickle",
        "group": "GNPS", "tag": "nist2gnps_cross"},
    6: {"lib": "/home/ausu/projects/yfs/data/NistData/nalib.pickle",
        "query": "/home/ausu/projects/yfs/data/NG_CROSS/GcorossN0.5_0.95.pickle",
        "group": "NIST", "tag": "gnps2nist_cross"},

    7: {"lib": "/home/ausu/projects/yfs/data/GNPSdata_f2/ob_train_ref.pickle",
        "query": "/home/ausu/projects/yfs/data/GNPSdata_f2/unique_gnps_sim_0.5_0.95.pickle",
        "group": "GNPS", "tag": "gnps_analog"},
    8: {"lib": "/home/ausu/projects/yfs/data/GNPSdata_f2/ob_train_ref.pickle",
        "query": "/home/ausu/projects/yfs/data/NistData/unique_nist_sim_0.5_0.95.pickle",
        "group": "GNPS", "tag": "nist2gnps_cross"},

    9: {"lib": "/home/ausu/projects/yfs/data/NistData/train_ref.pickle",
        "query": "/home/ausu/projects/yfs/data/NistData/N2N0.5_0.95.pickle",
        "group": "NIST", "tag": "nist_libsearch"},
    10: {"lib": "/home/ausu/projects/yfs/data/NistData/train_ref.pickle",
        "query": "/home/ausu/projects/yfs/data/NG_CROSS/G2N0.5_0.95.pickle",
        "group": "NIST", "tag": "gnps2nist_cross"},
}

MSDF = "/home/ausu/projects/yfs/MSDiffAugServer/ablition5"
MODEL_FILES = {
    "msdiffaug": {
        "GNPS": [f"/home/ausu/projects/yfs/MSDiffAugServer/ablition8/p/t100/MSDiffAug_126.pt",
                 # f"{MSDF}/msdiffaug/pth1/MSDiffAug_35.pt",
                 f"{MSDF}/msdiffaug/pth2/MSDiffAug_gnps.pt",
                 f"{MSDF}/msdiffaug/pth2/MSDiffAug_21.pt"],
        "NIST": ["/home/ausu/projects/yfs/MSDiffAugServer/ablition2/pth/MaskTop5"
                 "/MSDiffAug_25.pt"],
    },
    "msdiffaug_pr": {
        "GNPS": ["/home/ausu/projects/yfs/MSDiffAugServer/ablition2/pth/"
                 "MaskTop5PlusRandom1/MSDiffAug.pt"],
        "NIST": ["/home/ausu/projects/yfs/MSDiffAugServer/ablition2/pth/"
                 "MaskTop5PlusRandom1/MSDiffAug.pt"],
    },
    "msbert": {
        "GNPS": [f"{MSDF}/msbert/gpth/msbert_14.pt",
                 f"{MSDF}/msbert/pth2/msbert_7.pt",
                 f"{MSDF}/msbert/pth3/msbert_12.pt"],
        "NIST": [f"{MSDF}/msbert/N_pth1/msbert_8.pt",
                 f"{MSDF}/msbert/N_pth2/msbert_27.pt",
                 f"{MSDF}/msbert/N_pth3/msbert_23.pt"],
    },
    "ms2deepscore": {
        "GNPS": ["ms2deepscore_trained11.model",
                 "ms2deepscore_trained12.model",
                 "ms2deepscore_trained13.model"],
        "NIST": ["N_ms2deepscore_trained1.model",
                 "N_ms2deepscore_trained2.model",
                 "N_ms2deepscore_trained3.model"],
    },
    "spec2vec": {
        "GNPS": [f"{MSDF}/spe2vec/Spec2VecModel/ob_spec2vec.model",
                 f"{MSDF}/spe2vec/Spec2VecModel2/ob_spec2vec.model",
                 f"{MSDF}/spe2vec/Spec2VecModel3/ob_spec2vec.model"],
        "NIST": [f"{MSDF}/spe2vec/N_Spec2VecModel/ob_spec2vec.model",
                 f"{MSDF}/spe2vec/N_Spec2VecModel2/ob_spec2vec.model",
                 f"{MSDF}/spe2vec/N_Spec2VecModel3/ob_spec2vec.model"],
    },
    "msms": {},
}

# 旧脚本生成的 npz (NPZ_SOURCE_CHOICE = 1 时按 (模型, DATA_CHOICE) 查谱库 npz;
# 第二列查询 npz 已不再使用, 保留备查)
OLD_NPZ = {
    ("msdiffaug", 1): (f"{MSDF}/library1/MSDiffAug_gilib.npz",
                       f"{MSDF}/library1/MSDiffAug_gaq.npz"),
    ("msdiffaug", 2): (f"{MSDF}/library/MSDiffAug_galib.npz",
                       f"{MSDF}/library/MSDiffAug_gaq.npz"),
    ("msdiffaug", 3): (f"{MSDF}/N_library2/MSDiffAug_nilib.npz",
                       f"{MSDF}/N_library2/MSDiffAug_niq.npz"),
    ("msdiffaug", 4): (f"{MSDF}/N_library1/MSDiffAug_nAlib.npz",
                       f"{MSDF}/N_library1/MSDiffAug_naq.npz"),
    ("msdiffaug", 5): (f"{MSDF}/library/MSDiffAug_galib.npz",
                       f"{MSDF}/library/MSDiffAug_NcorossG.npz"),
    ("msdiffaug", 6): (f"{MSDF}/N_library1/MSDiffAug_nAlib.npz",
                       f"{MSDF}/N_library1/MSDiffAug_GcorossN.npz"),
    ("msbert", 1): (f"{MSDF}/library1/MSBERT_gilib.npz",
                    f"{MSDF}/library1/MSBERT_gaq.npz"),
    ("msbert", 2): (f"{MSDF}/library/MSBERT_galib.npz",
                    f"{MSDF}/library/MSBERT_gaq.npz"),
    ("msbert", 3): (f"{MSDF}/N_library3/MSBERT_nilib.npz",
                    f"{MSDF}/N_library3/MSBERT_niq.npz"),
    ("msbert", 4): (f"{MSDF}/N_library1/MSBERT_nAlib.npz",
                    f"{MSDF}/N_library1/MSBERT_naq.npz"),
    ("msbert", 5): (f"{MSDF}/library/MSBERT_galib.npz",
                    f"{MSDF}/library/MSBERT_NcorossG.npz"),
    ("msbert", 6): (f"{MSDF}/N_library1/MSBERT_nAlib.npz",
                    f"{MSDF}/N_library1/MSBERT_GcorossN.npz"),
    ("ms2deepscore", 1): (f"{MSDF}/library/MS2DeepScore_GNPS_Library.npz",
                          f"{MSDF}/library/MS2DeepScore_GNPS_query2.npz"),
    ("ms2deepscore", 2): (f"{MSDF}/library/MS2DeepScore_galib.npz",
                          f"{MSDF}/library3/MS2DeepScore_gaq.npz"),
    ("ms2deepscore", 3): (f"{MSDF}/N_library3/MS2DeepScore_nilib.npz",
                          f"{MSDF}/N_library3/MS2DeepScore_niq.npz"),
    ("ms2deepscore", 4): (f"{MSDF}/N_library1/MS2DeepScore_nalib.npz",
                          f"{MSDF}/N_library1/MS2DeepScore_naq.npz"),
    ("ms2deepscore", 5): (f"{MSDF}/library/MS2DeepScore_galib.npz",
                          f"{MSDF}/library3/MS2DeepScore_NcorossG.npz"),
    ("ms2deepscore", 6): (f"{MSDF}/N_library1/MS2DeepScore_nalib.npz",
                          f"{MSDF}/N_library1/MS2DeepScore_GcorossN.npz"),
    ("spec2vec", 1): (f"{MSDF}/library3/Spec2Vec_GNPS_Library.npz",
                      f"{MSDF}/library3/Spec2Vec_nist_query2.npz"),
    ("spec2vec", 2): (f"{MSDF}/library3/Spec2Vec_galib.npz",
                      f"{MSDF}/library3/Spec2Vec_gaq.npz"),
    ("spec2vec", 5): (f"{MSDF}/library3/Spec2Vec_galib.npz",
                      f"{MSDF}/library2/Spec2Vec_NcorossG.npz"),
    ("msms", 2): (f"{MSDF}/library/msms_galib.npz",
                  f"{MSDF}/library/msms_gaq.npz"),
    ("msms", 5): (f"{MSDF}/library/msms_galib.npz",
                  f"{MSDF}/library/msms_NcorossG.npz"),
}

# ================================================================
# 由上面的选择计算出的实际配置 (不要手动改)
# ================================================================

MODELS = {1: "msdiffaug", 2: "msbert", 3: "ms2deepscore", 4: "spec2vec",
          5: "msms", 6: "msdiffaug_pr"}
TASKS = {1: "library", 2: "analog", 3: "both"}
EMBED_MODES = {1: "from-scratch", 2: "load"}

MODEL = MODELS[MODEL_CHOICE]
TASK = TASKS[TASK_CHOICE]
EMBED_MODE = EMBED_MODES[EMBED_MODE_CHOICE]
DEVICE = "cuda" if DEVICE_CHOICE == 1 else "cpu"
HIT_MODE = "inchikey" if HIT_MODE_CHOICE == 1 else "smiles"

combo = DATA_COMBOS[DATA_CHOICE]
LIBRARY_DATA = combo["lib"]
QUERY_DATA = combo["query"]

if MODEL != "msms":
    files = MODEL_FILES[MODEL][combo["group"]]
    if not (1 <= MODEL_FILE_CHOICE <= len(files)):
        raise ValueError(
            f"MODEL_FILE_CHOICE={MODEL_FILE_CHOICE} 超出 {MODEL} 的 "
            f"{combo['group']} 权重数 ({len(files)} 个), 请改回 1~{len(files)}")
    MODEL_FILE = files[MODEL_FILE_CHOICE - 1]
else:
    MODEL_FILE = None

if NPZ_SOURCE_CHOICE == 0:
    LIB_NPZ = os.path.join(_HERE, "benchmark_npz",
                           f"{MODEL}_{combo['tag']}_lib.npz")
else:
    key = (MODEL, DATA_CHOICE)
    if key not in OLD_NPZ:
        available = sorted(c for m, c in OLD_NPZ if m == MODEL)
        raise ValueError(
            f"OLD_NPZ 表里没有 {MODEL} + 数据组合 {DATA_CHOICE} 的旧谱库 npz "
            f"({MODEL} 已有的组合: {available})。"
            f"请改用 NPZ_SOURCE_CHOICE = 0 或 EMBED_MODE_CHOICE = 1 重新嵌入")
    LIB_NPZ = OLD_NPZ[key][0]

# 查询嵌入 npz (服务器新版 test_benchmark.py 会保存/加载查询嵌入)
QUERY_NPZ = os.path.join(_HERE, "benchmark_npz",
                         f"{MODEL}_{combo['tag']}_query.npz")
