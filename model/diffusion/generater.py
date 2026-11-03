
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
for p in (_SCRIPT_DIR, _REPO_ROOT):
    if p not in sys.path:
        sys.path.append(p)

import time
import functools
import torch
import numpy as np
from tqdm import tqdm
import numpy as np
import torch
from tqdm import tqdm, trange
from diffsion_augment_model import DiffudionAugment
from gaussian_diffusion import SpacedDiffusion, space_timesteps
import gaussian_diffusion as gd
from utils.step_sample import TruncatedSampler
import torch.nn as nn
from info_nce import InfoNCE




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
        self.contrastive_loss = InfoNCE(0.05)

        schedule_tag = self.args.noise_schedule
        if self.args.beta_start is not None and self.args.beta_end is not None:
            schedule_tag += f"-{self.args.beta_start:g}-{self.args.beta_end:g}"
        self.save_dir = os.path.join(
            self.args.save_dir,
            f"mr{self.args.mask_ratio:g}_t{self.args.t_max}_{schedule_tag}")
        os.makedirs(self.save_dir, exist_ok=True)

        self._create_model()
        self._set_optimizer()
        self._load_checkpoint()
        # 检查点权重可能是 bf16/fp16: 与 fp16 autocast 混用时 DeBERTa 内 query/key 投影输出 fp32、
        # 注意力分数却是 half, masked_fill 写入 finfo(fp32).min 会溢出。统一回 fp32 主权重。
        self.model = self.model.float()
        self.diffusion = self.diffusion.float()
        # self._set_stopper()

    def _create_model(self):
        self.model = DiffudionAugment(self.device, self.args)
        self.model.to(self.device)

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
        self.schedule_sampler = TruncatedSampler(self.diffusion, t_max=self.args.t_max)

    def _set_optimizer(self):

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.args.learning_rate,
                                          # betas=(0.9, 0.999),
                                          weight_decay=self.args.weight_decay)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.args.use_fp16)

    def _checkpoint_dir(self):
        # 优先使用 --check_path 指定目录，否则用本次运行的 save_dir
        return self.args.check_path or self.save_dir

    def _list_checkpoints(self, ckpt_dir):
        # 扫描 checkpoint_epoch_X.pt，返回 {epoch: 路径}
        if not os.path.isdir(ckpt_dir):
            return {}
        found = {}
        prefix, suffix = "checkpoint_epoch_", ".pt"
        for f in os.listdir(ckpt_dir):
            if f.startswith(prefix) and f.endswith(suffix):
                try:
                    epoch = int(f[len(prefix):-len(suffix)])
                except ValueError:
                    continue
                found[epoch] = os.path.join(ckpt_dir, f)
        return found

    def _load_checkpoint(self):
        # 两种恢复方式：
        # 1) --resume_checkpoint latest|路径：完整恢复（模型+扩散+优化器+起始epoch），
        #    从保存的 epoch 之后继续训练
        # 2) --diffusion_epoch N：只加载第 N 个 epoch 的预训练扩散权重继续训练
        #    （keepon training，与 ablition4 trainner 的模式一致），优化器重新初始化
        ckpt_dir = self._checkpoint_dir()
        resume = str(self.args.resume_checkpoint)

        if resume != "none":
            if resume == "latest":
                checkpoints = self._list_checkpoints(ckpt_dir)
                if not checkpoints:
                    print(f"[resume] no checkpoint_epoch_*.pt found in {ckpt_dir}, start from scratch")
                    return
                ckpt_path = checkpoints[max(checkpoints)]
            else:
                ckpt_path = resume
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"[resume] checkpoint not found: {ckpt_path}")
            state = torch.load(ckpt_path, map_location='cpu')
            self.model.load_state_dict(state["model"])
            self.diffusion.load_state_dict(state["diffusion"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.start_epoch = int(state["epoch"]) + 1
            torch.cuda.empty_cache()
            print(f"[resume] restored from {ckpt_path}, continue from epoch {self.start_epoch}")
            return

        epoch = getattr(self.args, "diffusion_epoch", None)
        if epoch is None or epoch < 0:
            return
        model_file = os.path.join(ckpt_dir, f"diff_augment_epoch_{epoch}.pt")
        diffusion_file = os.path.join(ckpt_dir, f"spaced_diffusion_epoch_{epoch}.pt")
        if os.path.exists(model_file) and os.path.exists(diffusion_file):
            self.model.load_state_dict(torch.load(model_file, map_location='cpu'))
            self.diffusion.load_state_dict(torch.load(diffusion_file, map_location='cpu'))
            torch.cuda.empty_cache()
            print(f"[checkpoint] loaded pretrained diffusion weights of epoch {epoch} from {ckpt_dir}")
        else:
            print(f"[checkpoint] epoch {epoch} weights not found in {ckpt_dir}, start from scratch")

    def _train_one_epoch(self, epoch, only_bert_train=False, is_diffusion_train=True):

        tr_loss = 0
        tr_diff_loss = 0
        tr_mse_loss = 0
        tr_inten_loss = 0
        tr_mlm_loss = 0
        tr_cl_loss = 0
        train_time = []

        self.model.train()
        prog_iter = tqdm(self.train_dataloader, leave=False, desc='Training')

        for batch in prog_iter:
            train_start = time.time()

            #  input_idss输入序列、attention_mask掩码（padding位置为0，其余位置为1）、masked_indices0选择扩散的位置
            input_ids, attention_mask, masked_indices0, intensity = \
                batch["input_ids"].to(self.device, non_blocking=True), \
                    batch["attention_mask"].to(self.device, non_blocking=True), \
                    batch["masked_indices0"].to(self.device, non_blocking=True), \
                    batch["intensity"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            t, weights = self.schedule_sampler.sample(input_ids.shape[0], self.device)
            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.model,
                t,
                input_ids,
                masked_indices0.long(),
                attention_mask,
                intensity=intensity
            )
            #  此处为基于上下文感知扩散 (联合预测 m/z + 强度)
            # 禁用 autocast: transformers 4.56.1 modular DeBERTa 的 q_bias/v_bias 是独立 fp32 参数,
            # autocast 下会与 half 的 query 相加提升为 fp32, 而注意力分数 matmul 按 half 计算,
            # masked_fill 把 finfo(fp32).min 写进 half 直接溢出 (RuntimeError: value cannot be
            # converted to type at::Half without overflow)。扩散模型很小 (hidden 64/2层), 直接 fp32。
            with torch.cuda.amp.autocast(enabled=False):
                diff_mse_loss, diff_nll_loss, diff_inten_loss, aug_seq1, aug_seq2, aug_inten1, aug_inten2 = compute_losses()
                loss = diff_mse_loss + diff_nll_loss + self.args.intensity_loss_weight * diff_inten_loss

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            tr_diff_loss += diff_nll_loss.item() / len(self.train_dataloader)
            tr_mse_loss += diff_mse_loss.item() / len(self.train_dataloader)
            tr_inten_loss += diff_inten_loss.item() / len(self.train_dataloader)
            # tr_cl_loss += contrastive_loss / len(self.train_dataloader)
            tr_loss += loss.item() / len(self.train_dataloader)

            train_end = time.time()
            train_time.append(train_end - train_start)
        print(f' epoch {epoch}: mse_loss {tr_mse_loss:.4f}', end='   ')
        print(f'diff_loss {tr_diff_loss:.4f}', end='   ')
        print(f'inten_loss {tr_inten_loss:.4f}', end='   ')
        print(f'total_loss {tr_loss:.4f}')

        # **保存 SpacedDiffusion 模型的状态
        if epoch % 20 == 0:  # 每10个epoch保存一次
            print(f"Saving SpacedDiffusion model at epoch {epoch}...")
            diffusion_state_dict = self.diffusion.state_dict()
            diff_augment_state_dict = self.model.state_dict()
            torch.save(diffusion_state_dict,
                       os.path.join(self.save_dir, f"spaced_diffusion_epoch_{epoch}.pt"))
            torch.save(diff_augment_state_dict,
                       os.path.join(self.save_dir, f"diff_augment_epoch_{epoch}.pt"))
            # 完整训练状态（含优化器与 epoch），供 --resume_checkpoint 恢复
            torch.save({
                "epoch": epoch,
                "model": diff_augment_state_dict,
                "diffusion": diffusion_state_dict,
                "optimizer": self.optimizer.state_dict(),
            }, os.path.join(self.save_dir, f"checkpoint_epoch_{epoch}.pt"))

    def train(self):
        print("********** Running training **********")
        train_time = []

        for epoch in trange(self.start_epoch, self.start_epoch + int(self.args.epochs), desc="Epoch"):
            t = self._train_one_epoch(epoch, only_bert_train=False, is_diffusion_train=True)

            train_time.append(t)

            # if epoch % 10 == 0:
            #     self.eval(epoch, test=False)  #valid
            #
            #     self.eval(epoch, test=True)  #test
