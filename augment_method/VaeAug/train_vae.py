# -*- coding: utf-8 -*-
"""Stage 1: 训练 0.1 Da 密集 bin VAE (GNPS Orbitrap 训练集, 与 MSDiffAug 同数据)。

用法: python train_vae.py [--vae_epochs 100 ...]
检查点: vae_baseline/pth/vae_epoch_{e}.pt + vae_final.pt (stage 2 默认加载)
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
for p in (_HERE, _REPO_ROOT):
    if p not in sys.path:
        sys.path.append(p)

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import trange

from utils.diff_utils import set_seed
from utils.ms_utils import ParseOrbitrap
from data_process.process_data import MakeTrainData
from vae_model import DenseSpectraVAE, mu_prior_loss, spectra_to_dense
from utils.diff_config import get_config

TRAIN_PICKLE = '/home/ausu/projects/yfs/data/GNPSdata_f2/ob_train_ref.pickle'
SAVE_DIR = '/home/ausu/projects/yfs/MSDiffAugServer/vae_baseline/pth'


class VaeDataset(Dataset):
    def __init__(self, input_ids, intensity):
        self.input_ids = input_ids    # [N, 100] long
        self.intensity = intensity    # [N, 1, 100] float

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.intensity[idx]


def main():
    args = get_config()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    train_ref, msms1, precursor1, smiles1 = ParseOrbitrap(TRAIN_PICKLE)
    train_ref, word2idx = MakeTrainData(msms1, precursor1, 100)
    input_ids = torch.stack([torch.LongTensor(i[0]) for i in train_ref])
    intensity = torch.stack([torch.FloatTensor(i[1]) for i in train_ref])
    print(f'spectra: {len(input_ids)}')

    dataset = VaeDataset(input_ids, intensity)
    dataloader = DataLoader(dataset, batch_size=args.vae_batch_size, shuffle=True,
                            num_workers=4, pin_memory=True, persistent_workers=True)

    vae = DenseSpectraVAE(dim=args.vae_dim, hidden=args.vae_hidden, mid=args.vae_mid,
                          latent=args.vae_latent, sigma0=args.vae_sigma0).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=args.vae_lr)
    os.makedirs(SAVE_DIR, exist_ok=True)

    for epoch in trange(1, args.vae_epochs + 1, desc='VAE epoch'):
        vae.train()
        tr_recon = tr_peaks = tr_peakmu = tr_mu2 = tr_loss = 0.0
        for batch_ids, batch_inten in dataloader:
            batch_ids = batch_ids.to(device, non_blocking=True)
            batch_inten = batch_inten.to(device, non_blocking=True)
            dense = spectra_to_dense(batch_ids, batch_inten, args.vae_dim)
            optimizer.zero_grad(set_to_none=True)
            recon, mu = vae(dense)
            # 零 bin 大幅降权 + 峰 bin 加权: 稀疏谱零 bin 占 99%, 否则 MSE 被零 bin 主导
            w = args.vae_zero_weight + args.vae_peak_weight * dense
            loss_recon = (F.mse_loss(recon, dense, reduction='none') * w).mean()
            peak_mask = dense > 0
            loss_peaks = (F.mse_loss(recon[peak_mask], dense[peak_mask])
                          if peak_mask.any() else torch.tensor(0.0, device=device))
            # 确定性重构质量 (健康指标: 应随训练持续下降; 停滞=编码器没学到峰信息)
            with torch.no_grad():
                recon_mu = vae.decode(mu)
            loss_peakmu = (F.mse_loss(recon_mu[peak_mask], dense[peak_mask])
                           if peak_mask.any() else torch.tensor(0.0, device=device))
            loss_mu = mu_prior_loss(mu)
            loss = loss_recon + args.vae_mu_weight * loss_mu
            loss.backward()
            optimizer.step()
            n = len(dataloader)
            tr_recon += loss_recon.item() / n
            tr_peaks += loss_peaks.item() / n
            tr_peakmu += loss_peakmu.item() / n
            tr_mu2 += mu.pow(2).mean().item() / n
            tr_loss += loss.item() / n
        print(f'epoch {epoch}: recon {tr_recon:.5f} peaks {tr_peaks:.5f} '
              f'peaks@mu {tr_peakmu:.5f} mu2 {tr_mu2:.5f} total {tr_loss:.5f}')
        if epoch % args.vae_save_every == 0 or epoch == args.vae_epochs:
            torch.save(vae.state_dict(), os.path.join(SAVE_DIR, f'vae_epoch_{epoch}.pt'))
    torch.save(vae.state_dict(), os.path.join(SAVE_DIR, 'vae_final.pt'))
    print(f'saved to {SAVE_DIR}')


if __name__ == '__main__':
    main()
