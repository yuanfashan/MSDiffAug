import sys

sys.path.append('./')
import os
import time
import torch
from tqdm import tqdm, trange


import torch.nn as nn
from info_nce import InfoNCE
from utils.ms_utils import ParseOrbitrap, ModelEmbed, SearchTop_MemoryEfficient
from CAUG import mp_cm_model



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

        self.save_dir = self.args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self._create_model()
        self._set_optimizer()
        # self._set_stopper()

    def _create_model(self):
        self.msencoder = mp_cm_model(100002, 512, 6, 16, 0, 100, 3)
        self.msencoder.to(self.device)

    def _set_optimizer(self):

        self.optimizer = torch.optim.Adam(self.msencoder.parameters(),
                                          lr=1e-4,
                                          # betas=(0.9, 0.999),
                                          weight_decay=self.args.weight_decay)

    def _train_one_epoch(self, epoch, only_bert_train=False, is_diffusion_train=True):
        tr_loss = 0
        tr_diff_loss = 0
        tr_mlm_loss = 0
        tr_cl_loss = 0
        tr_mcl_loss = 0
        train_time = []

        self.msencoder.train()
        prog_iter = tqdm(self.train_dataloader, leave=False, desc='Training')

        for batch in prog_iter:
            train_start = time.time()

            input_ids, intensity = \
                    batch["input_ids"].to(self.device), \
                    batch["intensity"].to(self.device)

            self.optimizer.zero_grad()

            # 前向传播
            logits_lm1, logits_lm2, mask_token1, mask_token2, pool1, pool2= self.msencoder(input_ids, intensity)
            # 计算重建损失
            mlm_loss = self.criterion(logits_lm1.view(-1, logits_lm1.size(-1)),
                                                    mask_token1.view(-1))
            cl_loss = self.contrastive_loss(pool1.squeeze(), pool2.squeeze())
            loss = mlm_loss + cl_loss
            loss.backward()
            self.optimizer.step()

            tr_mlm_loss += loss / len(self.train_dataloader)
            tr_loss += loss.item() / len(self.train_dataloader)

            train_end = time.time()
            train_time.append(train_end - train_start)
        print(f' epoch {epoch}: mlm_loss {tr_mlm_loss:.4f}', end='   ')


    def train(self):
        print("********** Running training **********")
        train_time = []

        for epoch in trange(self.start_epoch, self.start_epoch + int(self.args.epochs), desc="Epoch"):
            t = self._train_one_epoch(epoch, only_bert_train=False, is_diffusion_train=True)

            train_time.append(t)


    def test_step(self,  epoch, best_accuracy, test_ref, test_query, smiles1, smiles2, batch_size=128):
        self.msencoder.eval()
        with torch.no_grad():
            test_ref_arr = ModelEmbed(self.msencoder, test_ref, batch_size)
            test_query_arr = ModelEmbed(self.msencoder, test_query, batch_size)
            top = SearchTop_MemoryEfficient(test_ref_arr, test_query_arr, smiles1, smiles2)
            top1 = top[0]
            pass
            # **保存 msencoder 模型的状态**

            if top1 > best_accuracy:
                best_accuracy = top1
                print(f"Saving msencoder model at epoch..")
                msencoder_state_dict = self.msencoder.state_dict()
                torch.save(msencoder_state_dict, os.path.join("/home/ausu/projects/yfs/MSDiffAugServer/ablition/pth/convaug", f"mp_cm_model_{epoch}.pt"))
            print(f"Test Top-1 Accuracy: {top1 * 100:.2f}%")
        return top1, best_accuracy

