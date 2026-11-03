# -*- coding: utf-8 -*-
"""VAE 谱图增强基线 (固定后验方差 σ₀ 的 σ-VAE, 后验采样生成增强视图)。

VAE 在 0.1 Da 密集 bin 向量上训练: 输入为 sqrt(归一化强度) 的 10000 维向量,
只含碎片峰 (前体离子位置 0 不参与)。增强视图 = 编码原谱后从后验 N(μ, σ₀²I)
采样再解码, 经相邻 bin 非极大抑制 + top-K 峰提取转回 token/强度格式, 与
ablition2 扩散增强的 aug_seq/aug_inten 输出格式一致。

为何固定 σ₀ 而非学习 σ: 稀疏谱 (0.1% 非零 bin) 重构损失数值极小 (~0.01-0.1),
而 N(0,I) 先验下 128 维 KL 天然 ~2-4 nats/维 (σ≠1 即付代价, σ→0 时发散),
两者差 3-4 个数量级 → 可学习 σ 的 VAE 必然塌缩到先验 N(0,I) (μ→0, σ→1),
β/free_bits 无法平衡。固定小 σ₀ 后 KL 的梯度承载部分化为 0.5·‖μ‖²,
训练退化为加噪 AE + 轻 L2 正则, 结构上不可能塌缩; 增强视图是可控尺度的
同谱微扰 (正样本对语义保持)。
"""
import torch
import torch.nn as nn

MIN_MZ_TOKEN = 2
MIN_FRAG_BIN = 150  # 最小碎片质量 15.0 Da 对应的 0.1 Da bin


def mu_prior_loss(mu):
    """固定后验方差 σ₀²I 时 KL(N(μ,σ₀²I) || N(0,I)) 的梯度承载部分 = 0.5·‖μ‖²。"""
    return 0.5 * mu.pow(2).sum(dim=1).mean()


class DenseSpectraVAE(nn.Module):
    def __init__(self, dim=10000, hidden=1024, mid=512, latent=128, sigma0=0.1):
        super().__init__()
        self.dim = dim
        self.sigma0 = sigma0
        self.encoder = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, mid), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(mid, latent)
        self.decoder = nn.Sequential(
            nn.Linear(latent, mid), nn.ReLU(),
            nn.Linear(mid, hidden), nn.ReLU(),
            nn.Linear(hidden, dim),
        )

    def encode(self, x):
        return self.fc_mu(self.encoder(x))

    def reparameterize(self, mu):
        return mu + self.sigma0 * torch.randn_like(mu)

    def decode(self, z):
        return torch.sigmoid(self.decoder(z))

    def forward(self, x):
        mu = self.encode(x)
        z = self.reparameterize(mu)
        return self.decode(z), mu


def spectra_to_dense(input_ids, intensity, dim=10000):
    """token 谱 [B,L] -> 0.1 Da 密集向量 [B,dim] (sqrt 归一化强度, 前体离子除外)。

    token = round(m/z,2)*100 + 2, 故 bin = (token-2)//10 = floor(m/z*10), 整数运算无浮点误差。
    """
    if intensity.dim() == 3:
        intensity = intensity.squeeze(1)
    B, L = input_ids.shape
    device = input_ids.device
    is_peak = input_ids >= MIN_MZ_TOKEN
    is_peak[:, 0] = False  # 前体离子不参与 VAE
    rows = torch.arange(B, device=device).unsqueeze(1).expand(B, L)[is_peak]
    bins = ((input_ids[is_peak] - MIN_MZ_TOKEN) // 10).clamp(0, dim - 1)
    vals = intensity[is_peak].clamp_min(0.0).sqrt()
    dense = torch.zeros(B, dim, device=device)
    dense = dense.index_put_((rows, bins), vals, accumulate=True).clamp(0.0, 1.0)
    return dense


class SpectraVAEAugmenter:
    """冻结 VAE 的增强视图生成器: 后验采样 -> 解码 -> 峰提取 -> token/强度输出。

    峰提取: 相邻 bin 非极大抑制 (宽峰拖尾只留局部极大), 每谱取与原谱碎片峰
    数量相同的 top-K (上限 99), 按 m/z 升序回填; 前体离子位置保持干净。
    """

    def __init__(self, vae, dim=10000, max_len=100, sample_temp=1.0):
        self.vae = vae
        self.dim = dim
        self.max_len = max_len
        self.sample_temp = sample_temp

    @torch.no_grad()
    def augment_view(self, input_ids, intensity, crop_precursor=True):
        """单视图: 后验采样解码 -> 峰提取 (兼容旧接口)。"""
        return self.augment_views(input_ids, intensity, n_views=1,
                                  crop_precursor=crop_precursor)[0]

    @torch.no_grad()
    def augment_views(self, input_ids, intensity, n_views=2, crop_precursor=True):
        """n 个增强视图: 只编码一次, n 次后验采样解码。

        相比循环调用 n 次 augment_view, 省掉 n-1 次 [B,10000] 编码 (MLP 三层)。
        """
        if intensity.dim() == 3:
            intensity = intensity.squeeze(1)
        B, L = input_ids.shape

        dense = spectra_to_dense(input_ids, intensity, self.dim)
        mu = self.vae.encode(dense)
        zs = [mu + self.sample_temp * self.vae.sigma0 * torch.randn_like(mu)
              for _ in range(n_views)]
        y_all = self.vae.decode(torch.cat(zs, dim=0)).square()  # [n*B, dim] 一次解码
        views = []
        for k in range(n_views):
            views.append(self._extract_peaks(y_all[k * B:(k + 1) * B],
                                             input_ids, intensity, crop_precursor))
        return views

    def _extract_peaks(self, y, input_ids, intensity, crop_precursor):
        B, L = input_ids.shape
        device = input_ids.device
        K = self.max_len - 1

        if crop_precursor:
            # 前体裁剪: 解码器无前体条件, 会在 >前体 / <15 Da 区间生成非物理峰
            prec_bin = ((input_ids[:, 0] - MIN_MZ_TOKEN) // 10).clamp(0, self.dim - 1)
            bins = torch.arange(self.dim, device=device).unsqueeze(0)
            unphysical = (bins > prec_bin.unsqueeze(1)) | (bins < MIN_FRAG_BIN)
            y = torch.where(unphysical, torch.zeros_like(y), y)

        # 相邻 bin 非极大抑制
        pad = torch.zeros(B, 1, device=device)
        is_max = (y >= torch.cat([pad, y[:, :-1]], dim=1)) & \
                 (y >= torch.cat([y[:, 1:], pad], dim=1))
        y_nms = torch.where(is_max, y, torch.zeros_like(y))

        n_peaks = (input_ids[:, 1:] >= MIN_MZ_TOKEN).sum(dim=1).clamp(max=K)  # [B]
        sorted_vals, sorted_bins = torch.topk(y_nms, K, dim=1)  # [B, K] 降序
        rank = torch.arange(K, device=device).unsqueeze(0)  # [1, K]
        keep = (rank < n_peaks.unsqueeze(1)) & (sorted_vals > 0)  # [B, K]
        kept_bins = sorted_bins[keep]
        kept_vals = sorted_vals[keep]
        counts = keep.sum(dim=1)                      # [B]
        offsets = torch.cumsum(counts, dim=0) - counts
        total = int(counts.sum().item())

        aug_seq = torch.zeros(B, L, dtype=torch.long, device=device)
        aug_inten = torch.zeros(B, L, dtype=torch.float32, device=device)
        aug_seq[:, 0] = input_ids[:, 0]               # 前体保持干净
        aug_inten[:, 0] = intensity[:, 0]

        if total > 0:
            flat_pos = torch.arange(total, device=device)
            rows = torch.searchsorted(offsets, flat_pos, right=True) - 1
            # 按 (行, bin 升序) 排序, 键唯一 (bins < dim+1)
            key = rows * (self.dim + 1) + kept_bins
            order = key.argsort()
            rows_s = rows[order]
            cols_s = (flat_pos - offsets[rows_s]) + 1  # 行内按排序后枚举位置回填
            aug_seq[rows_s, cols_s] = kept_bins[order] * 10 + MIN_MZ_TOKEN
            aug_inten[rows_s, cols_s] = kept_vals[order].clamp(0.0, 1.0)
        return aug_seq, aug_inten
