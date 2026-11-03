import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (_HERE, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

import csv
import json
import math
import time
import functools
import random
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from scipy.stats import wilcoxon

from model.diffusion.diffsion_augment_model import DiffudionAugment
from model.diffusion.gaussian_diffusion import SpacedDiffusion, space_timesteps
from model.diffusion import gaussian_diffusion as gd
from utils.ms_utils import ParseOrbitrap
from utils.step_sample import UniformSampler
from data_process.process_data import MakeTrainData
from augment_method.VaeAug.vae_model import DenseSpectraVAE, SpectraVAEAugmenter, MIN_MZ_TOKEN

import torch.nn as nn

# 单点维护的硬编码路径 (合并前化学合理性/峰级两文件夹各一份)
DATA_PICKLE = '/home/ausu/projects/yfs/data/GNPSdata_f2/ob_train_ref.pickle'
MODEL_FILE = '/home/ausu/projects/yfs/MSDiffAugServer/ablition8/pthln/mr0.1_t1000_cosine-1e-05-0.001/diff_augment_epoch_100.pt'
DIFFUSION_FILE = '/home/ausu/projects/yfs/MSDiffAugServer/ablition8/pthln/mr0.1_t1000_cosine-1e-05-0.001/spaced_diffusion_epoch_100.pt'
VAE_FILE = '/home/ausu/projects/yfs/MSDiffAugServer/vae_baseline/pth/vae_final.pt'
# 推断采样温度 (实际噪声 = σ₀ × temp): 0.1 = 近模态解码, 提升重构保真度 (可调)
VAE_SAMPLE_TEMP = 0.1
# 评估循环批大小 (原 64): 调大摊薄 per-batch 固定开销 (VAE 编码/指标调度), 显存允许可再调
QUERY_BATCH_SIZE = 128


class MyDataSet(Dataset):
    def __init__(self, input_ids, intensity, mask_ratio=0.1, intensity_bin=None):
        '''
        初始化数据集，接收两个输入参数：input_ids 和 intensity

        :param input_ids: 输入 ID，形状为 (样本数, 100)
        :param intensity: 强度值，形状为 (样本数, 100)
        :param mask_ratio: 用于随机掩蔽的位置比例（默认为 0.1）
        :param intensity_bin: 可选，强度等级标签（与 input_ids 等长），用于峰级一致性分组
        '''
        self.input_ids = input_ids  # 记录 input_ids
        self.intensity = intensity  # 记录 intensity
        self.mask_ratio = mask_ratio  # 掩蔽的比例
        self.intensity_bin = intensity_bin

    def __len__(self):
        '''
        返回数据集的大小（即样本数）
        '''
        return len(self.input_ids)

    def __getitem__(self, idx):
        '''
        根据索引返回样本的 input_ids, intensity, attention_mask 和 masked_indices0，返回字典格式

        :param idx: 样本的索引
        :return: 一个字典，包含 'input_ids', 'intensity', 'attention_mask' 和 'masked_indices0'
        '''
        # 获取当前样本的 input_ids 和 intensity
        input_ids = torch.tensor(self.input_ids[idx], dtype=torch.long)
        intensity = torch.tensor(self.intensity[idx], dtype=torch.float)

        # 生成 attention_mask，input_id 为 0 时 mask 为 0.0，其他为 1.0
        attention_mask = (input_ids != 0).float()

        # 生成 masked_indices0，按 mask_ratio 的比率随机标记非零位置为 1
        # 获取所有非零位置的索引，前体（位置 0）不参与扰动
        non_zero_indices = (input_ids != 0).nonzero(as_tuple=True)[0]
        non_zero_indices = non_zero_indices[non_zero_indices > 0]

        # num_masked = min(max(int(len(non_zero_indices) * self.mask_ratio), 2), len(non_zero_indices))
        x = max(int(len(non_zero_indices) * 0.1), 1)
        num_masked = max(int(round(5 * math.log(x) / math.log(10))), 1)
        if num_masked > 0:
            masked_indices = np.random.choice(non_zero_indices.numpy(), num_masked, replace=False)
        else:
            masked_indices = np.array([], dtype=np.int64)

        # 创建 masked_indices0，初始全为 0
        masked_indices0 = torch.zeros_like(input_ids, dtype=torch.bool)

        # 将随机选择的位置标记为 1
        masked_indices0[masked_indices] = 1

        result = {
            'input_ids': input_ids,
            'intensity': intensity,
            'attention_mask': attention_mask,
            'masked_indices0': masked_indices0
        }
        if self.intensity_bin is not None:
            result['intensity_bin'] = torch.tensor(self.intensity_bin[idx], dtype=torch.long)
        return result


class DataGenerator2:
    def __init__(self, max_len=100):
        # 超参数
        self.max_len = max_len

        # 加载数据
        train_ref, msms1, precursor1, smiles1 = ParseOrbitrap(DATA_PICKLE)
        print(len(smiles1))
        train_ref, word2idx = MakeTrainData(msms1, precursor1, self.max_len)
        self.train_data = train_ref  # MakeTrainData 产物，评估时作为谱库与增广来源
        self.smiles = smiles1

        # 词汇表大小
        self.vocab_size = len(word2idx)
        self.word2idx = word2idx

        # 准备数据集
        input_ids, intensity, _, _ = zip(*train_ref)
        intensity = [torch.FloatTensor(i) for i in intensity]
        input_ids = [torch.LongTensor(i) for i in input_ids]

        # 创建数据集
        self.dataset = MyDataSet(input_ids, intensity)

        # 生成数据加载器
        self.train_dataloader = DataLoader(self.dataset, batch_size=64, shuffle=True)
        self.valid_dataloader = DataLoader(self.dataset, batch_size=64, shuffle=False)  # 可以根据需要修改
        self.test_dataloader = DataLoader(self.dataset, batch_size=64, shuffle=False)  # 可以根据需要修改

        # 数据集的大小
        self.item_size = len(self.dataset)

    def get_dataloaders(self):
        return self.train_dataloader, self.valid_dataloader, self.test_dataloader


def _accumulate_chem(store, r, sample_idx):
    """把单条谱的化学合理性计数累加进方法级汇总 store。"""
    store["n_gen"] += r["n_gen"]
    store["c1_ok"] += r["c1_ok"]
    store["c1_low"] += r["c1_low"]
    store["c1_high"] += r["c1_high"]
    store["c2_ok"] += r["c2_ok"]
    store["n_chg_low"] += r["n_chg_low"]
    store["n_chg_med"] += r["n_chg_med"]
    store["n_chg_high"] += r["n_chg_high"]
    store["c2_ok_low"] += r["c2_ok_low"]
    store["c2_ok_med"] += r["c2_ok_med"]
    store["c2_ok_high"] += r["c2_ok_high"]
    store["c3_ok"] += r["c3_ok"]
    store["c3_nl"].extend(r["c3_nl"])
    store["c7_ok"] += r["c7_ok"]
    store["n_chg"] += r["n_chg"]
    store["c4_ok"] += r["c4_ok"]
    store["c4_nl"].extend(r["c4_nl"])
    for om, gm, kind in r["violations"]:
        store["violations"].append((sample_idx, r["precursor_mz"], om, gm, kind, gm - om))


def _wilcoxon_tests(rows, out_dir):
    """ISC (含强度分层)/PNR/SNL/FNL 的 Wilcoxon 符号秩检验 (方法对, 按谱配对)。

    rows: [(指标名, 方法A名, A 每谱命中率, 方法B名, B 每谱命中率), ...]
    每谱双视图聚合为单一命中率; 双侧/单侧 (greater/less) p 值一并报告,
    p 下溢为 0 时显示为 <1e-300。各指标打印在同一张表, 保存为 wilcoxon_tests.csv。
    """
    def _pfmt(p):
        return "<1e-300" if p == 0.0 else f"{p:.4g}"

    print("\n显著性检验: Wilcoxon 符号秩检验, 按谱配对 (每谱聚合双视图)")
    header = (f"{'cmp':<18}{'metric':<8}{'N_pairs':>10}{'W(two)':>12}{'W+(gt)':>12}"
              f"{'p_two':>12}{'p_gt':>12}{'p_lt':>12}{'mean_a':>12}{'mean_b':>12}")
    print(header)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "wilcoxon_tests.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cmp", "metric", "method_a", "method_b", "n_pairs",
                         "W_two_sided", "W_greater", "p_two_sided", "p_greater",
                         "p_less", "mean_a", "mean_b"])
        for name, label_a, rate_a, label_b, rate_b in rows:
            if len(rate_a) > 0:
                w_two, p_two = wilcoxon(rate_a, rate_b, alternative="two-sided")
                w_gt, p_gt = wilcoxon(rate_a, rate_b, alternative="greater")
                _, p_lt = wilcoxon(rate_a, rate_b, alternative="less")
                mean_a = float(np.mean(rate_a))
                mean_b = float(np.mean(rate_b))
                print(f"{label_a + ' vs ' + label_b:<18}{name:<8}{len(rate_a):>10}"
                      f"{w_two:>12.1f}{w_gt:>12.1f}"
                      f"{_pfmt(p_two):>12}{_pfmt(p_gt):>12}{_pfmt(p_lt):>12}"
                      f"{mean_a:>12.4f}{mean_b:>12.4f}")
                writer.writerow([f"{label_a} vs {label_b}", name, label_a, label_b,
                                 len(rate_a), f"{w_two:.1f}", f"{w_gt:.1f}",
                                 _pfmt(p_two), _pfmt(p_gt), _pfmt(p_lt),
                                 f"{mean_a:.4f}", f"{mean_b:.4f}"])
            else:
                print(f"{label_a + ' vs ' + label_b:<18}{name:<8}{'0':>10}"
                      f"{'-':>12}{'-':>12}{'-':>12}{'-':>12}{'-':>12}"
                      f"{'-':>12}{'-':>12}{'-':>12}")
                writer.writerow([f"{label_a} vs {label_b}", name, label_a, label_b,
                                 0, "nan", "nan", "nan", "nan", "nan", "nan", "nan"])


class Trainer:
    def __init__(self, args, device, generator):

        self.args = args
        self.device = device
        self.start_epoch = 0  # define the start epoch for keepon trainingzhonss

        self.loss_func = torch.nn.BCEWithLogitsLoss()
        self.generator = generator
        self.train_dataloader = generator.train_dataloader
        self.valid_dataloader = generator.valid_dataloader
        self.test_dataloader = generator.test_dataloader
        self.item_size = 100002
        self.args.item_size = 100002
        self.generator = generator
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        self._create_model()

    def _create_model(self):
        self.model = DiffudionAugment(self.device, self.args)
        self.model.to(self.device)
        self.model.load_state_dict(torch.load(MODEL_FILE))

        betas = gd.get_named_beta_schedule(self.args.noise_schedule, self.args.diffusion_steps,
                                           beta_start=self.args.beta_start,
                                           beta_end=self.args.beta_end)
        timestep_respacing = [self.args.diffusion_steps]
        self.diffusion = SpacedDiffusion(
            use_timesteps=space_timesteps(self.args.diffusion_steps, timestep_respacing),
            betas=betas,
            rescale_timesteps=self.args.rescale_timesteps,
            predict_xstart=self.args.predict_xstart,
            learn_sigmas=self.args.learn_sigma,
            sigma_small=self.args.sigma_small,
            use_kl=self.args.use_kl,
            rescale_learned_sigmas=self.args.rescale_learned_sigmas
        )
        self.diffusion.load_state_dict(torch.load(DIFFUSION_FILE))
        self.schedule_sampler = UniformSampler(self.diffusion)

        # 冻结 VAE (stage 1 默认超参): 增强视图 = 后验采样重构
        vae = DenseSpectraVAE(dim=10000, hidden=1024, mid=512, latent=128, sigma0=0.1)
        vae.load_state_dict(torch.load(VAE_FILE, map_location='cpu'))
        vae.to(self.device)
        vae.eval()
        for p in vae.parameters():
            p.requires_grad_(False)
        self.vae_augmenter = SpectraVAEAugmenter(vae, dim=10000, max_len=100, sample_temp=VAE_SAMPLE_TEMP)

    def _train_one_epoch(self, epoch, only_bert_train=False, is_diffusion_train=True):
        tr_loss = 0
        tr_diff_loss = 0
        tr_mlm_loss = 0
        tr_cl_loss = 0
        tr_mcl_loss = 0
        train_time = []

        self.model.eval()
        self.diffusion.eval()
        prog_iter = tqdm(self.train_dataloader, leave=False, desc='Training')

        for batch in prog_iter:
            train_start = time.time()

            #  input_idss输入序列、attention_mask掩码（padding位置为0，其余位置为1）、masked_indices0选择扩散的位置
            input_ids, attention_mask, masked_indices0, intensity = \
                    batch["input_ids"].to(self.device), \
                    batch["attention_mask"].to(self.device), \
                    batch["masked_indices0"].to(self.device), \
                    batch["intensity"].to(self.device)

            t, weights = self.schedule_sampler.sample(input_ids.shape[0], self.device)
            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.model,
                t,
                input_ids,
                masked_indices0.long(),
                attention_mask
            )
            #  此处为基于上下文感知扩散
            _, _, aug_seq1, aug_seq2 = compute_losses()

    def _random_augment(self, input_ids, masked_indices0):
        """随机基线增广: 与扩散共用相同 mask 位置, token id 均匀偏移 ±500 (即 ±5 Da)。"""
        mask = masked_indices0.bool()
        views = []
        for _ in range(2):
            offsets = torch.randint(-500, 501, input_ids.shape, device=input_ids.device)
            views.append(torch.where(mask, (input_ids + offsets).clamp(2, 100001), input_ids))
        return views[0], views[1]

    def _vae_augment(self, input_ids, intensity, masked_indices0, bin_align=False):
        """VAE 基线增广, 锚点钉回干净谱。

        bin_align=False: 按序回填, masked 位置用 VAE 同序号生成峰 (化学合理性用,
        保留 VAE 生成峰的完整分布, SNL/FNL 才有意义)。
        bin_align=True: bin 级对齐 (峰级一致性用), 无容差窗口。VAE 输出量化在
        0.1 Da bin 网格上, "保留" = VAE 在原峰所在 bin 内有输出峰 (同 bin 判定);
        masked 位置回填该 bin 的峰 token, 同 bin 无峰记峰消失 (0)。
        """
        mask_bool = masked_indices0.bool()
        views = []
        for seq, _ in self.vae_augmenter.augment_views(input_ids, intensity, n_views=2):
            if not bin_align:
                views.append(torch.where(mask_bool, seq, input_ids))
                continue
            cand = seq.clone()
            cand[:, 0] = 0  # 前体位置不作为候选峰
            target = (input_ids - MIN_MZ_TOKEN) // 10 * 10 + MIN_MZ_TOKEN  # 原峰所在 bin 的峰 token
            has = (cand[:, None, :] == target[:, :, None]).any(dim=2)      # [B,L] 同 bin 有峰
            aligned = torch.where(mask_bool,
                                  torch.where(has, target, torch.zeros_like(target)),
                                  input_ids)
            views.append(aligned)
        return views[0], views[1]

    @staticmethod
    def _accumulate(store, m, group_names):
        for g in group_names:
            pres, tot = m["conserved"][g]
            store["conserved"][g][0] += pres
            store["conserved"][g][1] += tot
            tpres, ttot = m["conserved_tol"][g]
            store["conserved_tol"][g][0] += tpres
            store["conserved_tol"][g][1] += ttot
        chg, tot = m["masked_changed_counts"]
        store["masked_changed"][0] += chg
        store["masked_changed"][1] += tot
        new, tot = m["new_peak_counts"]
        store["new_peak_counts"][0] += new
        store["new_peak_counts"][1] += tot
        hits, tot = m["isc_counts"]
        store["isc"][0] += hits
        store["isc"][1] += tot
        store["new_peak_ratios"].append(m["new_peak_ratio"])
        store["weighted_cosine"].append(m["weighted_cosine"])
        store["masked_cosine"].append(m["masked_cosine"])

    def test_step(self, epoch, best_accuracy, ref_data, query_data, smiles_ref, smiles_query):
        """峰级一致性评估: 原谱 / 扩散增广 / 随机增广 / VAE 增广 对比。

        query_data 为增广来源 (原谱)；对每个 masked 位置生成 diff/random/vae 视图
        (vae 视图为 bin 级对齐: 同 0.1 Da bin 判定保留, 无同 bin 峰记峰消失), 统计保留率
        (精确匹配 CPR 与 0.02 Da 容差 CPR.02, 按强度分组)、新峰率、m/z 位移分布、
        余弦相似度。结果保存到本目录 results/peak_level/ 下。
        """
        from metrics_utils import (ids_to_vec, vec_cosine, spectrum_metrics,
                                   is_common_neutral_loss, GROUP_NAMES)

        # ---------- 数据准备 ----------
        query_ids, query_inten, _, query_bin = zip(*query_data)
        query_inten = [np.asarray(i, dtype=np.float32).reshape(-1) for i in query_inten]
        query_bin = [np.asarray(b, dtype=np.int64).reshape(-1) for b in query_bin]

        # mask 比例与训练数据保持一致 (用户在 DataGenerator2 中设置)
        mask_ratio = getattr(self.train_dataloader.dataset, "mask_ratio", 0.1)
        print(f"Peak-level consistency: mask_ratio={mask_ratio}, query={len(query_ids)}")

        query_dataset = MyDataSet(list(query_ids), list(query_inten),
                                  mask_ratio=mask_ratio, intensity_bin=list(query_bin))
        query_loader = DataLoader(query_dataset, batch_size=QUERY_BATCH_SIZE, shuffle=False)

        self.model.eval()
        self.diffusion.eval()

        # ---------- 生成增强并逐谱统计 ----------
        agg = {m: {"conserved": {g: [0, 0] for g in GROUP_NAMES},
                   "conserved_tol": {g: [0, 0] for g in GROUP_NAMES},
                   "masked_changed": [0, 0],
                   "new_peak_counts": [0, 0],
                   "new_peak_ratios": [],
                   "isc": [0, 0],
                   "weighted_cosine": [], "masked_cosine": [],
                   "view_consistency": []} for m in ("original", "diff", "random", "vae")}
        dev_rows = []   # (method, delta_da, intensity_bin, orig_id, aug_id)
        records = []    # 逐样本记录 -> results/records.jsonl

        sample_idx = 0
        timing = {"diff": 0.0, "vae": 0.0, "metrics": 0.0}
        prog = tqdm(query_loader, leave=False, desc="Augmenting query")
        for batch in prog:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            masked_indices0 = batch["masked_indices0"].to(self.device)
            intensity = batch["intensity"].reshape(input_ids.shape[0], -1).to(self.device)
            inten_bin = batch["intensity_bin"]

            bsz = input_ids.shape[0]
            batch_smiles = list(smiles_query[sample_idx: sample_idx + bsz])
            sample_idx += bsz

            # 迭代反向链 (DDIM, eta=0) 生成两个增强视图, 锚点位置保持干净
            # 禁用 autocast: transformers 4.56.1 modular DeBERTa 的 fp32 q_bias/v_bias 在 autocast
            # 下与 half query 相加会提升为 fp32, masked_fill 写 finfo(fp32).min 进 half 溢出
            t0 = time.time()
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=False):
                    diff1, _ = self.diffusion.augment_view_ddim(
                        self.model, input_ids, masked_indices0.long(), attention_mask,
                        intensity=intensity,
                        ddim_steps=self.args.ddim_sampling_steps,
                        t_start=self.args.t_max - 1)
                    diff2, _ = self.diffusion.augment_view_ddim(
                        self.model, input_ids, masked_indices0.long(), attention_mask,
                        intensity=intensity,
                        ddim_steps=self.args.ddim_sampling_steps,
                        t_start=self.args.t_max - 1)
            timing["diff"] += time.time() - t0
            rand1, rand2 = self._random_augment(input_ids, masked_indices0)
            t0 = time.time()
            vae1, vae2 = self._vae_augment(input_ids, intensity, masked_indices0,
                                           bin_align=True)
            timing["vae"] += time.time() - t0
            t0 = time.time()

            orig_np = input_ids.cpu().numpy()
            inten_np = intensity.cpu().numpy()
            bin_np = inten_bin.cpu().numpy()
            mask_np = masked_indices0.cpu().numpy()
            diff1_np, diff2_np = diff1.cpu().numpy(), diff2.cpu().numpy()
            rand1_np, rand2_np = rand1.cpu().numpy(), rand2.cpu().numpy()
            vae1_np, vae2_np = vae1.cpu().numpy(), vae2.cpu().numpy()

            for i in range(bsz):
                orig = [int(x) for x in orig_np[i]]
                inten = [float(x) for x in inten_np[i]]
                bins = [int(x) for x in bin_np[i]]
                masked = [p for p in range(len(orig)) if mask_np[i][p] == 1]

                # 原谱 (参照)
                m = spectrum_metrics(orig, inten, orig, [])
                self._accumulate(agg["original"], m, GROUP_NAMES)
                agg["original"]["view_consistency"].append(1.0)

                record = {"idx": sample_idx - bsz + i, "smiles": batch_smiles[i],
                          "precursor_mz": round((orig[0] - 2) * 0.01, 2),
                          "orig_ids": orig, "intensity": inten, "bins": bins,
                          "masked": masked}

                for method, (v1, v2) in (("diff", (diff1_np[i], diff2_np[i])),
                                         ("random", (rand1_np[i], rand2_np[i])),
                                         ("vae", (vae1_np[i], vae2_np[i]))):
                    ids1 = [int(x) for x in v1]
                    ids2 = [int(x) for x in v2]
                    record[method + "_view1"] = ids1
                    record[method + "_view2"] = ids2
                    for ids_v in (ids1, ids2):
                        m = spectrum_metrics(orig, inten, ids_v, masked,
                                             bin_mode=(method == "vae"))
                        self._accumulate(agg[method], m, GROUP_NAMES)
                        for p in m["masked_changed"]:
                            if ids_v[p] != 0:
                                dev_rows.append((method, (ids_v[p] - orig[p]) * 0.01,
                                                 bins[p], orig[p], ids_v[p]))
                    agg[method]["view_consistency"].append(
                        vec_cosine(ids_to_vec(ids1, inten), ids_to_vec(ids2, inten)))
                records.append(record)
            timing["metrics"] += time.time() - t0

        n_batches = max(len(query_loader), 1)
        print(f"\n[timing] diff {timing['diff']:.1f}s | vae {timing['vae']:.1f}s | "
              f"metrics {timing['metrics']:.1f}s | 每批: "
              f"diff {timing['diff'] / n_batches * 1000:.0f}ms, "
              f"vae {timing['vae'] / n_batches * 1000:.0f}ms, "
              f"metrics {timing['metrics'] / n_batches * 1000:.0f}ms")

        # ---------- 汇总 ----------
        summary = {}
        for method in ("original", "diff", "random", "vae"):
            store = agg[method]
            row = {}
            for g in GROUP_NAMES:
                pres, tot = store["conserved"][g]
                row["conserved_" + g] = float(pres / tot) if tot > 0 else float("nan")
            # 容差保留率 CPR.02: |m' - m| <= 0.02 Da 即视为保留 (跨强度组汇总)
            tpres = sum(store["conserved_tol"][g][0] for g in GROUP_NAMES)
            ttot = sum(store["conserved_tol"][g][1] for g in GROUP_NAMES)
            row["conserved_tol"] = float(tpres / ttot) if ttot > 0 else float("nan")
            # 总 Conserved rate = 非锚点峰中被保留的峰总数 / 非锚点峰总数
            chg, tot = store["masked_changed"]
            row["conserved"] = 1.0 - float(chg / tot) if tot > 0 else float("nan")
            new, tot = store["new_peak_counts"]
            row["new_peak_ratio"] = float(new / tot) if tot > 0 else float("nan")
            hits, tot = store["isc"]
            row["isc"] = float(hits / tot) if tot > 0 else float("nan")
            ratios = store["new_peak_ratios"]
            row["new_peak_ratio_max"] = float(max(ratios)) if ratios else float("nan")
            row["new_peak_ratio_min"] = float(min(ratios)) if ratios else float("nan")
            row["weighted_cosine"] = float(np.mean(store["weighted_cosine"]))
            mcos_vals = [v for v in store["masked_cosine"] if not math.isnan(v)]
            row["masked_cosine"] = float(np.mean(mcos_vals)) if mcos_vals else float("nan")
            row["view_consistency"] = float(np.mean(store["view_consistency"]))
            devs = [d for mtd, d, *_ in dev_rows if mtd == method]
            if devs:
                darr = np.asarray(devs, dtype=np.float64)  # 带符号 δ = m' - m (Da)
                row["mz_dev_mean"] = float(np.mean(np.abs(darr)))
                row["mz_dev_median"] = float(np.median(np.abs(darr)))
                row["mz_dev_q25"] = float(np.quantile(darr, 0.25))
                row["mz_dev_q75"] = float(np.quantile(darr, 0.75))
                row["mz_dev_iqr"] = row["mz_dev_q75"] - row["mz_dev_q25"]
                row["mz_dev_maxabs"] = float(np.max(np.abs(darr)))
                row["mz_dev_le005"] = float(np.mean(np.abs(darr) <= 0.05))
                row["mz_dev_le01"] = float(np.mean(np.abs(darr) <= 0.10))
            else:
                row.update(mz_dev_mean=float("nan"), mz_dev_median=float("nan"),
                           mz_dev_q25=float("nan"), mz_dev_q75=float("nan"),
                           mz_dev_iqr=float("nan"), mz_dev_maxabs=float("nan"),
                           mz_dev_le005=float("nan"), mz_dev_le01=float("nan"))
            summary[method] = row

        # ---------- 打印汇总 ----------
        print("\n" + "=" * 140)
        print("Peak-level consistency (原谱 original / 扩散增广 diff / 随机增广 random / VAE 增广 vae)")
        print("=" * 140)

        def _f(v, w=10, nd=3):
            return f"{v:>{w}.{nd}f}" if not math.isnan(v) else f"{'-':>{w}}"

        print("\n保留率 CPR (n_C/n_R, 修改后 m/z 与原谱精确一致的重构峰占比; CPR.02 = |m'-m|<=0.02 Da; "
              "vae 为 bin 级对齐: 保留 = 同 0.1 Da bin 有输出峰):")
        print(f"{'method':<10}{'CPR':>9}{'CPR.02':>9}{'C_High':>9}{'C_Med':>9}{'C_Low':>9}")
        for method in ("original", "diff", "random", "vae"):
            row = summary[method]
            line = f"{method:<10}" + _f(row["conserved"], 9, 3) + _f(row["conserved_tol"], 9, 3)
            for g in GROUP_NAMES:
                line += _f(row["conserved_" + g], 9, 3)
            print(line)

        print("\n新峰率 NPR (n_N/n_R, 修改后 m/z 发生变化的重构峰占比, CPR+NPR=1; vae: 同 bin 无峰记消失; 含各样本最大/最小):")
        print(f"{'method':<10}{'NewPeak':>9}{'Max':>9}{'Min':>9}")
        for method in ("original", "diff", "random", "vae"):
            row = summary[method]
            line = f"{method:<10}" + _f(row["new_peak_ratio"], 9, 3)
            line += _f(row["new_peak_ratio_max"], 9, 3)
            line += _f(row["new_peak_ratio_min"], 9, 3)
            print(line)

        print("\nISC 谱内一致性 (n_ISC/n_N, 新生成峰落在其他原有谱峰 ±0.2 Da 内的比例, 消失峰计为未命中; "
              "vae 非保留峰均记消失, 故 ISC 恒为 0):")
        print(f"{'method':<10}{'ISC':>9}")
        for method in ("original", "diff", "random", "vae"):
            row = summary[method]
            print(f"{method:<10}" + _f(row["isc"], 9, 3))

        print("\nm/z 位移分布 (被改且未消失峰, 单位 Da; Mean/Med 为 |δ|, 50% QI 为带符号 δ 分位数):")
        print(f"{'method':<10}{'|δ|Mean':>9}{'|δ|Med':>8}{'|δ|<=0.05%':>12}{'|δ|<=0.1%':>11}"
              f"{'50% QI [Q25,Q75]':>20}{'IQR':>8}{'Max|δ|':>9}")
        for method in ("original", "diff", "random", "vae"):
            row = summary[method]
            line = f"{method:<10}"
            if math.isnan(row["mz_dev_mean"]):
                line += f"{'-':>9}{'-':>8}{'-':>12}{'-':>11}{'-':>20}{'-':>8}{'-':>9}"
            else:
                line += _f(row["mz_dev_mean"], 9, 3)
                line += _f(row["mz_dev_median"], 8, 3)
                line += _f(row["mz_dev_le005"] * 100, 12, 1)
                line += _f(row["mz_dev_le01"] * 100, 11, 1)
                line += f"[{row['mz_dev_q25']:>8.3f},{row['mz_dev_q75']:>8.3f}]"
                line += _f(row["mz_dev_iqr"], 8, 3)
                line += _f(row["mz_dev_maxabs"], 9, 3)
            print(line)

        print("\n余弦相似度 (原谱 vs 生成谱整谱 / masked位置 / 两视图间):")
        print(f"{'method':<10}{'W_Cos':>8}{'W_CosM':>9}{'ViewCos':>8}")
        for method in ("original", "diff", "random", "vae"):
            row = summary[method]
            line = f"{method:<10}"
            line += f"{row['weighted_cosine']:>8.3f}"
            line += f"{row['masked_cosine']:>9.3f}" if not math.isnan(row["masked_cosine"]) else f"{'-':>9}"
            line += f"{row['view_consistency']:>8.3f}"
            print(line)

        # ---------- 保存结果 ----------
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "peak_level")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            cols = (["method", "conserved", "conserved_tol"] + ["conserved_" + g for g in GROUP_NAMES] +
                    ["new_peak_ratio", "new_peak_ratio_max", "new_peak_ratio_min",
                     "isc",
                     "weighted_cosine", "masked_cosine", "view_consistency",
                     "mz_dev_mean", "mz_dev_median", "mz_dev_q25", "mz_dev_q75",
                     "mz_dev_iqr", "mz_dev_maxabs", "mz_dev_le005", "mz_dev_le01"])
            writer.writerow(cols)
            for method, row in summary.items():
                writer.writerow([method] + [row[c] for c in cols[1:]])

        # F(δ): 带符号位移 δ 的经验累积分布函数 (ECDF, 非统计 F 分布), 供画曲线
        with open(os.path.join(out_dir, "dev_ecdf.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["method", "delta_da", "F_delta"])
            for method in ("diff", "random", "vae"):
                devs = np.asarray([d for m, d, *_ in dev_rows if m == method])
                if devs.size == 0:
                    continue
                grid = np.linspace(devs.min(), devs.max(), 512)
                for x in grid:
                    writer.writerow([method, f"{x:.4f}", f"{float(np.mean(devs <= x)):.6f}"])

        with open(os.path.join(out_dir, "deviations.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["method", "delta_da", "abs_delta_da", "intensity_bin",
                             "orig_id", "aug_id", "is_common_neutral_loss"])
            for method, d, b, oid, aid in dev_rows:
                writer.writerow([method, f"{d:.4f}", f"{abs(d):.4f}", b, oid, aid,
                                 int(is_common_neutral_loss(d))])

        with open(os.path.join(out_dir, "records.jsonl"), "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

        print(f"结果已保存: {out_dir}")
        return 0.0, best_accuracy

    def evaluate_chemical_plausibility(self, query_data):
        """化学合理性评估: 原谱对照 (original) / 扩散增广 (diff) / 随机偏移 (random) / VAE 增广 (vae)。

        query_data 为增广来源 (原谱)；对每个 masked 位置生成 diff/random/vae 视图，
        在五项化学合理性指标上检查生成峰: PRC 前体质量约束 / ISC 谱内一致性 /
        PNR 前体中性丢失 / SNL 位移即中性丢失 / FNL 碎片中性丢失。
        ISC 进一步按原谱峰强度升序三等分 (Low/Med/High) 分层,
        考察 ISC 随强度的变化。
        original 行以原谱全部真实碎片峰作上限参照。
        并对 ISC (含 Low/Med/High 分层)/PNR/SNL/FNL 做 diff vs random 与
        vae vs random 的 Wilcoxon 符号秩检验 (按谱配对, 每谱聚合双视图的命中率)。
        结果保存到本目录 results/ 下。
        """
        from metrics_utils import (chemical_plausibility, COMMON_NEUTRAL_LOSSES,
                                   MZ_TOL, MIN_FRAG_MZ, ADDUCT_TOL)

        # ---------- 数据准备 ----------
        query_ids, query_inten, _, _ = zip(*query_data)
        query_inten = [np.asarray(i, dtype=np.float32).reshape(-1) for i in query_inten]

        # mask 比例与训练数据保持一致 (用户在 DataGenerator2 中设置)
        mask_ratio = getattr(self.train_dataloader.dataset, "mask_ratio", 0.1)
        print(f"Chemical plausibility: mask_ratio={mask_ratio}, query={len(query_ids)}")

        query_dataset = MyDataSet(list(query_ids), list(query_inten), mask_ratio=mask_ratio)
        query_loader = DataLoader(query_dataset, batch_size=QUERY_BATCH_SIZE, shuffle=False)

        self.model.eval()
        self.diffusion.eval()

        # ---------- 生成增广视图并逐谱检查五项化学合理性指标 ----------
        agg = {m: {"n_gen": 0, "c1_ok": 0, "c1_low": 0, "c1_high": 0,
                   "c2_ok": 0, "c3_ok": 0, "c3_nl": [], "c7_ok": 0,
                   "n_chg": 0, "c4_ok": 0, "c4_nl": [],
                   "n_chg_low": 0, "n_chg_med": 0, "n_chg_high": 0,
                   "c2_ok_low": 0, "c2_ok_med": 0, "c2_ok_high": 0,
                   "violations": []} for m in ("original", "diff", "random", "vae")}

        rate_store = {m: {"diff": [], "random_diff": [], "vae": [], "random_vae": []} for m in
                      ("isc", "isc_low", "isc_med", "isc_high", "pnr", "snl", "fnl")}

        sample_idx = 0
        timing = {"diff": 0.0, "vae": 0.0, "metrics": 0.0}
        prog = tqdm(query_loader, leave=False, desc="Augmenting query")
        for batch in prog:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            masked_indices0 = batch["masked_indices0"].to(self.device)
            intensity = batch["intensity"].to(self.device)

            bsz = input_ids.shape[0]
            # 迭代反向链 (DDIM, eta=0) 生成两个增强视图, 锚点位置保持干净
            # 禁用 autocast: transformers 4.56.1 modular DeBERTa 的 fp32 q_bias/v_bias 在 autocast
            # 下与 half query 相加会提升为 fp32, masked_fill 写 finfo(fp32).min 进 half 溢出
            t0 = time.time()
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=False):
                    diff1, _ = self.diffusion.augment_view_ddim(
                        self.model, input_ids, masked_indices0.long(), attention_mask,
                        intensity=intensity,
                        ddim_steps=self.args.ddim_sampling_steps,
                        t_start=self.args.t_max - 1)
                    diff2, _ = self.diffusion.augment_view_ddim(
                        self.model, input_ids, masked_indices0.long(), attention_mask,
                        intensity=intensity,
                        ddim_steps=self.args.ddim_sampling_steps,
                        t_start=self.args.t_max - 1)
            timing["diff"] += time.time() - t0
            rand1, rand2 = self._random_augment(input_ids, masked_indices0)
            t0 = time.time()
            vae1, vae2 = self._vae_augment(input_ids, intensity, masked_indices0)
            timing["vae"] += time.time() - t0
            t0 = time.time()

            orig_np = input_ids.cpu().numpy()
            mask_np = masked_indices0.cpu().numpy()
            inten_np = batch["intensity"].cpu().numpy()
            diff1_np, diff2_np = diff1.cpu().numpy(), diff2.cpu().numpy()
            rand1_np, rand2_np = rand1.cpu().numpy(), rand2.cpu().numpy()
            vae1_np, vae2_np = vae1.cpu().numpy(), vae2.cpu().numpy()

            for i in range(bsz):
                orig = [int(x) for x in orig_np[i]]
                masked = [int(p) for p in range(len(orig)) if mask_np[i][p] == 1]

                # 原谱对照: 全部真实碎片峰参与检查, 作为五项指标的上限参照
                all_peaks = [int(p) for p in range(1, len(orig)) if orig[p] != 0]
                r = chemical_plausibility(orig, orig, all_peaks, intensity=inten_np[i])
                _accumulate_chem(agg["original"], r, sample_idx + i)

                # 每谱双视图聚合的命中率, 供按谱配对的 Wilcoxon 检验
                sample_rates = {m: {} for m in ("diff", "random", "vae")}
                for method, (v1, v2) in (("diff", (diff1_np[i], diff2_np[i])),
                                         ("random", (rand1_np[i], rand2_np[i])),
                                         ("vae", (vae1_np[i], vae2_np[i]))):
                    c2_sum = c3_sum = c4_sum = c7_sum = 0
                    c2l_sum = c2m_sum = c2h_sum = 0
                    nchg_sum = ngen_sum = 0
                    nchl_sum = nchm_sum = nchh_sum = 0
                    for ids_v in (v1, v2):
                        r = chemical_plausibility(orig, [int(x) for x in ids_v], masked,
                                                  intensity=inten_np[i])
                        _accumulate_chem(agg[method], r, sample_idx + i)
                        c2_sum += r["c2_ok"]
                        c3_sum += r["c3_ok"]
                        c4_sum += r["c4_ok"]
                        c7_sum += r["c7_ok"]
                        nchg_sum += r["n_chg"]
                        ngen_sum += r["n_gen"]
                        c2l_sum += r["c2_ok_low"]; nchl_sum += r["n_chg_low"]
                        c2m_sum += r["c2_ok_med"]; nchm_sum += r["n_chg_med"]
                        c2h_sum += r["c2_ok_high"]; nchh_sum += r["n_chg_high"]
                    if nchg_sum > 0:
                        sample_rates[method]["isc"] = c2_sum / nchg_sum
                        sample_rates[method]["snl"] = c4_sum / nchg_sum
                    if ngen_sum > 0:
                        sample_rates[method]["pnr"] = c3_sum / ngen_sum
                        sample_rates[method]["fnl"] = c7_sum / ngen_sum
                    if nchl_sum > 0:
                        sample_rates[method]["isc_low"] = c2l_sum / nchl_sum
                    if nchm_sum > 0:
                        sample_rates[method]["isc_med"] = c2m_sum / nchm_sum
                    if nchh_sum > 0:
                        sample_rates[method]["isc_high"] = c2h_sum / nchh_sum
                # 配对收集: 只在该方法与该谱 random 命中率都存在时入列 (保证同谱配对),
                # 每个方法对各自维护 random 列, 避免共用列表导致长度错位
                for metric in ("isc", "isc_low", "isc_med", "isc_high",
                               "pnr", "snl", "fnl"):
                    r = sample_rates["random"].get(metric)
                    for method in ("diff", "vae"):
                        v = sample_rates[method].get(metric)
                        if v is not None and r is not None:
                            rate_store[metric][method].append(v)
                            rate_store[metric]["random_" + method].append(r)
            sample_idx += bsz
            timing["metrics"] += time.time() - t0

        n_batches = max(len(query_loader), 1)
        print(f"\n[timing] diff {timing['diff']:.1f}s | vae {timing['vae']:.1f}s | "
              f"metrics {timing['metrics']:.1f}s | 每批: "
              f"diff {timing['diff'] / n_batches * 1000:.0f}ms, "
              f"vae {timing['vae'] / n_batches * 1000:.0f}ms, "
              f"metrics {timing['metrics'] / n_batches * 1000:.0f}ms")

        # ---------- 汇总 ----------
        summary = {}
        for method in ("original", "diff", "random", "vae"):
            s = agg[method]
            n_gen, n_chg = s["n_gen"], s["n_chg"]
            summary[method] = {
                "n_gen": n_gen,
                "n_chg": n_chg,
                "prc": s["c1_ok"] / n_chg if n_chg else float("nan"),
                "prc_too_low": s["c1_low"] / n_chg if n_chg else float("nan"),
                "prc_too_high": s["c1_high"] / n_chg if n_chg else float("nan"),
                "isc": s["c2_ok"] / n_chg if n_chg else float("nan"),
                "n_chg_low": s["n_chg_low"],
                "n_chg_med": s["n_chg_med"],
                "n_chg_high": s["n_chg_high"],
                "isc_low": s["c2_ok_low"] / s["n_chg_low"] if s["n_chg_low"] else float("nan"),
                "isc_med": s["c2_ok_med"] / s["n_chg_med"] if s["n_chg_med"] else float("nan"),
                "isc_high": s["c2_ok_high"] / s["n_chg_high"] if s["n_chg_high"] else float("nan"),
                "pnr": s["c3_ok"] / n_gen if n_gen else float("nan"),
                "snl": s["c4_ok"] / n_chg if n_chg else float("nan"),
                "fnl": s["c7_ok"] / n_gen if n_gen else float("nan"),
            }

        # ---------- 打印汇总 ----------
        METHODS = ("original", "diff", "random", "vae")
        print("\n" + "=" * 120)
        print("化学合理性 Chemical plausibility (原谱 original 对照 / 扩散 diff / 随机 random / VAE vae)")
        print("=" * 120)

        def _f(v, w=12, nd=3):
            return f"{v:>{w}.{nd}f}" if not math.isnan(v) else f"{'-':>{w}}"

        print(f"\nPRC 前体质量约束: {MIN_FRAG_MZ:.2f} Da <= m' <= precursor + {ADDUCT_TOL:.2f} Da (新生成峰)")
        print(f"{'method':<10}{'n_chg':>10}{'PRC%':>10}{'TooLow%':>10}{'TooHigh%':>10}")
        for method in METHODS:
            row = summary[method]
            print(f"{method:<10}{row['n_chg']:>10}"
                  + _f(row["prc"] * 100, 10, 1)
                  + _f(row["prc_too_low"] * 100, 10, 2)
                  + _f(row["prc_too_high"] * 100, 10, 2))

        print(f"\nISC 谱内一致性: m' 落在原谱碎片集合 S 的 ±{MZ_TOL:.2f} Da 内 (新生成峰)")
        print(f"{'method':<10}{'n_chg':>10}{'ISC%':>10}")
        for method in METHODS:
            row = summary[method]
            print(f"{method:<10}{row['n_chg']:>10}"
                  + _f(row["isc"] * 100, 10, 2))

        print(f"\nISC 分层 (原谱峰按强度升序三等分): 原峰属于该层的新生成峰的 m' 落在 S 的 ±{MZ_TOL:.2f} Da 内")
        print(f"{'method':<10}{'n_Low':>8}{'ISC_Low%':>10}{'n_Med':>8}{'ISC_Med%':>10}"
              f"{'n_High':>8}{'ISC_High%':>10}")
        for method in METHODS:
            row = summary[method]
            print(f"{method:<10}{row['n_chg_low']:>8}" + _f(row["isc_low"] * 100, 10, 2)
                  + f"{row['n_chg_med']:>8}" + _f(row["isc_med"] * 100, 10, 2)
                  + f"{row['n_chg_high']:>8}" + _f(row["isc_high"] * 100, 10, 2))

        print(f"\nPNR 前体中性丢失: m' ≈ precursor - NL_j (±{MZ_TOL:.2f} Da) (生成峰)")
        print(f"{'method':<10}{'n_gen':>10}{'PNR%':>10}")
        for method in METHODS:
            row = summary[method]
            print(f"{method:<10}{row['n_gen']:>10}"
                  + _f(row["pnr"] * 100, 10, 2))

        print(f"\nSNL 位移即中性丢失: |m - m'| ≈ NL_j (±{MZ_TOL:.2f} Da) (新生成峰)")
        print(f"{'method':<10}{'n_chg':>10}{'SNL%':>10}")
        for method in METHODS:
            row = summary[method]
            print(f"{method:<10}{row['n_chg']:>10}"
                  + _f(row["snl"] * 100, 10, 2))

        print(f"\nFNL 碎片中性丢失: m' ≈ s - NL_j (s ∈ S), 落在 S' = S - NL 的 ±{MZ_TOL:.2f} Da 内 (生成峰)")
        print(f"{'method':<10}{'n_gen':>10}{'FNL%':>10}")
        for method in METHODS:
            row = summary[method]
            print(f"{method:<10}{row['n_gen']:>10}"
                  + _f(row["fnl"] * 100, 10, 2))

        # ---------- 显著性检验: Wilcoxon 符号秩 (diff/vae vs random, 按谱配对) ----------
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        metric_pairs = [("ISC", "isc"), ("ISC_L", "isc_low"), ("ISC_M", "isc_med"),
                        ("ISC_H", "isc_high"), ("PNR", "pnr"), ("SNL", "snl"),
                        ("FNL", "fnl")]
        wil_rows = []
        for name, key in metric_pairs:
            wil_rows.append((name, "diff", rate_store[key]["diff"],
                             "random", rate_store[key]["random_diff"]))
            wil_rows.append((name, "vae", rate_store[key]["vae"],
                             "random", rate_store[key]["random_vae"]))
        _wilcoxon_tests(wil_rows, out_dir)

        # ---------- 保存结果 ----------
        with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            cols = ["method", "n_gen", "n_chg", "prc", "prc_too_low", "prc_too_high",
                    "isc", "isc_low", "isc_med", "isc_high",
                    "pnr", "snl", "fnl"]
            writer.writerow(cols)
            for method, row in summary.items():
                writer.writerow([method] + [row[c] for c in cols[1:]])

        with open(os.path.join(out_dir, "neutral_loss_breakdown.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["method", "constraint", "nl_da", "count", "ratio"])
            for method in ("original", "diff", "random", "vae"):
                s = agg[method]
                for nl in COMMON_NEUTRAL_LOSSES:
                    n3 = s["c3_nl"].count(nl)
                    n4 = s["c4_nl"].count(nl)
                    if n3:
                        writer.writerow([method, "PNR_precursor_nl", nl, n3,
                                         n3 / s["n_gen"] if s["n_gen"] else 0.0])
                    if n4:
                        writer.writerow([method, "SNL_displacement_nl", nl, n4,
                                         n4 / s["n_chg"] if s["n_chg"] else 0.0])

        with open(os.path.join(out_dir, "violations.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["method", "sample_idx", "precursor_mz",
                             "orig_mz", "gen_mz", "kind", "delta_da"])
            for method in ("original", "diff", "random", "vae"):
                for idx, pmz, om, gm, kind, delta in agg[method]["violations"]:
                    writer.writerow([method, idx, f"{pmz:.2f}", f"{om:.2f}",
                                     f"{gm:.2f}", kind, f"{delta:.2f}"])

        print(f"结果已保存: {out_dir}")
        return summary
