import sys
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
for p in (_SCRIPT_DIR, _REPO_ROOT):
    if p not in sys.path:
        sys.path.append(p)

import time
import torch
from tqdm import tqdm, trange

# from model.diffusion.diffsion_augment_model import DiffudionAugment
from model.diffusion.diffsion_augment_model import DiffudionAugment
from model.diffusion.gaussian_diffusion import SpacedDiffusion, space_timesteps
from model.diffusion import gaussian_diffusion as gd
from utils.step_sample import UniformSampler, TruncatedSampler


import torch.nn as nn
from info_nce import InfoNCE
from utils.ms_utils import ParseOrbitrap, ModelEmbed, SearchTop_MemoryEfficient, SearchTopInChIKey_MemoryEfficient
from model.representation.msdiffaug import MSDiffAug_MaskTop5PlusRandom


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

        self._create_model()
        self._set_optimizer()
        # self._set_stopper()

    def _create_model(self):
        self.model = DiffudionAugment(self.device, self.args)
        self.model.to(self.device)
        model_file = '/home/ausu/projects/yfs/MSDiffAugServer/ablition8/nistpth/mr0.1_t500_cosine-1e-05-0.001/diff_augment_epoch_100.pt'
        # self.model.load_state_dict(torch.load(model_file))
        self.model.load_state_dict(torch.load(model_file, map_location='cpu'))

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
        model_file = '/home/ausu/projects/yfs/MSDiffAugServer/ablition8/nistpth/mr0.1_t500_cosine-1e-05-0.001/spaced_diffusion_epoch_100.pt'
        # self.diffusion.load_state_dict(torch.load(model_file))
        self.diffusion.load_state_dict(torch.load(model_file, map_location='cpu'))
        # self.schedule_sampler = UniformSampler(self.diffusion)
        self.schedule_sampler = TruncatedSampler(self.diffusion, t_max=self.args.t_max)
        self.msencoder = MSDiffAug_MaskTop5PlusRandom(100002, 512, 6, 16, 0.1, 100, 3)
        self.msencoder.to(self.device)

    def _set_optimizer(self):

        self.optimizer = torch.optim.Adam(self.msencoder.parameters(),
                                          lr=1e-4,
                                          # betas=(0.9, 0.999),
                                          weight_decay=self.args.weight_decay)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.args.use_fp16)

    def _train_one_epoch(self, epoch, only_bert_train=False, is_diffusion_train=True):
        tr_loss = 0
        tr_diff_loss = 0
        tr_mlm_loss = 0
        tr_cl_loss = 0
        tr_mcl_loss = 0
        train_time = []

        self.model.eval()
        self.diffusion.eval()
        self.msencoder.train()
        prog_iter = tqdm(self.train_dataloader, leave=False, desc='Training')

        for batch in prog_iter:
            train_start = time.time()

            #  input_idss输入序列、attention_mask掩码（padding位置为0，其余位置为1）、masked_indices0/1选择扩散的位置
            input_ids, attention_mask, masked_indices0, masked_indices1, intensity = \
                    batch["input_ids"].to(self.device, non_blocking=True), \
                    batch["attention_mask"].to(self.device, non_blocking=True), \
                    batch["masked_indices0"].to(self.device, non_blocking=True), \
                    batch["masked_indices1"].to(self.device, non_blocking=True), \
                    batch["intensity"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            #  迭代反向链 (DDIM, eta=0) 生成增强视图: 从 t_max-1 逐步去噪到 0, 锚点位置保持干净
            #  禁用 autocast: transformers 4.56.1 modular DeBERTa 的 fp32 q_bias/v_bias 在 autocast
            #  下与 half query 相加会提升为 fp32, masked_fill 写 finfo(fp32).min 进 half 溢出
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=False):
                    aug_seq, aug_inten = self.diffusion.augment_view_ddim(
                        self.model, input_ids, masked_indices0.long(), attention_mask,
                        intensity=intensity,
                        ddim_steps=self.args.ddim_sampling_steps,
                        t_start=self.args.t_max - 1)


            # 前向传播 (编码器接收增强视图的 token + 增强强度)
            with torch.cuda.amp.autocast(enabled=self.args.use_fp16):
                logits_lm1, logits_lm2, mask_token1, pool1, pool2= self.msencoder(
                    input_ids, aug_seq, intensity, aug_inten)
                # 计算重建损失
                mlm_loss = self.criterion(logits_lm1.view(-1, logits_lm1.size(-1)),
                                                        mask_token1.view(-1))
                cl_loss = self.contrastive_loss(pool1.squeeze(1), pool2.squeeze(1))
                loss = mlm_loss + cl_loss
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            tr_mlm_loss += mlm_loss.item() / len(self.train_dataloader)
            tr_cl_loss += cl_loss.item() / len(self.train_dataloader)
            tr_loss += loss.item() / len(self.train_dataloader)

            train_end = time.time()
            train_time.append(train_end - train_start)
        # if epoch %10==0:
        #     print("aug_seq1", aug_seq1[masked_indices0])
        #     print("aug_seq2", aug_seq2[masked_indices0])
        print(f' epoch {epoch}: mlm_loss {tr_mlm_loss:.4f}', end='   ')
        print(f'cl_loss {tr_cl_loss:.4f}', end='   ')
        print(f'total_loss {tr_loss:.4f}')


    def train(self):
        print("********** Running training **********")
        train_time = []

        for epoch in trange(self.start_epoch, self.start_epoch + int(self.args.epochs), desc="Epoch"):
            t = self._train_one_epoch(epoch, only_bert_train=False, is_diffusion_train=True)

            train_time.append(t)


    def test_step(self,  epoch, best_accuracy, test_ref, test_query, smiles1, smiles2, ik1=None, ik2=None, batch_size=128):
        self.msencoder.eval()
        with torch.no_grad():
            test_ref_arr = ModelEmbed(self.msencoder, test_ref, batch_size)
            test_query_arr = ModelEmbed(self.msencoder, test_query, batch_size)
            if ik1 is not None and ik2 is not None:
                top = SearchTopInChIKey_MemoryEfficient(test_ref_arr, test_query_arr, ik1, ik2)
            else:
                top = SearchTop_MemoryEfficient(test_ref_arr, test_query_arr, smiles1, smiles2)
            top1 = top[0]
            pass
            # **保存 msencoder 模型的状态**

            # torch.save(msencoder_state_dict, f"./pth/mp_cm_model_{epoch}.pt")
            if top1 > best_accuracy:
                best_accuracy = top1
                print(f"Saving msencoder model at epoch..")
                msencoder_state_dict = self.msencoder.state_dict()
                save_dir = "/home/ausu/projects/yfs/MSDiffAugServer/ablition8/p/t100"
                os.makedirs(save_dir, exist_ok=True)
                torch.save(msencoder_state_dict, os.path.join(save_dir, f"MSDiffAug_{epoch}.pt"))
            print(f"Test Top-1 Accuracy: {top1 * 100:.2f}%")
        return top1, best_accuracy

