# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 08:49:25 2023

@author: Administrator
"""
import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import torch.utils.data as Data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MyDataSet(Data.Dataset):
    def __init__(self, input_ids, intensity):
        self.input_ids = input_ids
        self.intensity = intensity

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.intensity[idx]


class Mask(nn.Module):
    def __init__(self, max_pred):
        super(Mask, self).__init__()
        self.max_pred = max_pred

    def forward(self, x, x1, intensity_):
        intensity_ = intensity_.squeeze(1)
        mask_bool_list = []
        mask_token_list = []
        mask_pos_list = []
        for p in range(x.shape[0]):
            pos_p = torch.argsort(intensity_[p, :], descending=True)  # 获取按强度排序的位置
            len_mask = 2  # 控制遮蔽的数量
            mask_pos = pos_p[torch.randperm(5)[0:len_mask]].to(device)  # 随机选择要遮蔽的位置
            mask_bool = torch.rand(x.shape[1]) > 1  # 生成随机的遮蔽布尔值
            mask_bool[mask_pos] = True  # 设置需要遮蔽的位置为True
            mask_bool = mask_bool.to(device)  # 将mask_bool移动到相应的设备上
            mask_bool_list.append(mask_bool.unsqueeze(0))

            mask_token = x[p, :][mask_pos]  # 获取原始x中对应位置的token

            if len_mask <= self.max_pred:
                n_pad = self.max_pred - len_mask
                mask_pos = torch.cat((mask_pos, torch.zeros(n_pad).long().to(device)), dim=0)
                mask_token = torch.cat((mask_token, torch.zeros(n_pad).long().to(device)), dim=0)

            mask_token_list.append(mask_token.unsqueeze(0))
            mask_pos_list.append(mask_pos.unsqueeze(0))

        mask_bool = torch.cat(mask_bool_list, dim=0)

        # 对x和x1应用相同的mask
        mask_x = torch.masked_fill(x, mask_bool, 1)  # 对x进行遮蔽
        mask_x1 = torch.masked_fill(x1, mask_bool, 1)  # 对x1进行遮蔽

        mask_token_list = torch.cat(mask_token_list, dim=0)
        mask_pos_list = torch.cat(mask_pos_list, dim=0)
        mask_token_list = mask_token_list.long()
        mask_pos_list = mask_pos_list.long()

        return mask_x, mask_x1, mask_token_list, mask_pos_list


class Mask2(nn.Module):
    def __init__(self, max_pred):
        super(Mask2, self).__init__()
        self.max_pred = max_pred

    def forward(self, x, intensity_):
        # 与原逐样本循环语义一致: 强度前5的峰中不放回随机选2个掩蔽, 并补齐到 max_pred;
        # 仅改为整批张量操作, 消除循环内的 CPU->GPU 拷贝与多次小 kernel 启动
        intensity_ = intensity_.squeeze(1)
        top5 = torch.argsort(intensity_, dim=1, descending=True)[:, :5]       # (B, 5)
        sel = torch.multinomial(torch.ones(x.shape[0], 5, device=x.device),
                                2, replacement=False)                         # (B, 2)
        mask_pos = torch.gather(top5, 1, sel)                                 # (B, 2)

        mask_bool = torch.zeros_like(x, dtype=torch.bool)
        mask_bool[torch.arange(x.shape[0], device=x.device)[:, None], mask_pos] = True
        mask_x = torch.masked_fill(x, mask_bool, 1)

        mask_pos_full = torch.zeros(x.shape[0], self.max_pred, dtype=torch.long, device=x.device)
        mask_pos_full[:, :2] = mask_pos
        mask_token = torch.gather(x, 1, mask_pos)                             # (B, 2)
        mask_token_full = torch.zeros(x.shape[0], self.max_pred, dtype=torch.long, device=x.device)
        mask_token_full[:, :2] = mask_token

        return mask_x, mask_token_full, mask_pos_full


class MaskTop5PlusRandom(nn.Module):
    """掩码变体: 强度 top5 的有效峰中随机掩 1 个, 其余有效峰中再随机掩 1 个
    (与 ablition2 的 MaskTop5PlusRandom 语义一致); 返回与 Mask2 相同的三元组。"""

    def __init__(self, max_pred):
        super(MaskTop5PlusRandom, self).__init__()
        self.max_pred = max_pred

    def forward(self, x, intensity_):
        intensity_ = intensity_.squeeze(1)
        mask_bool_list = []
        mask_token_list = []
        mask_pos_list = []
        for p in range(x.shape[0]):
            pos_p = torch.argsort(intensity_[p, :], descending=True)  # 按强度排序的位置
            valid_pos = (x[p, :] != 0).nonzero(as_tuple=True)[0]  # 所有有效位置
            if valid_pos.numel() == 0:
                valid_pos = torch.zeros(1, dtype=torch.long, device=x.device)
            top5_pool = pos_p[:5]  # 强度前5的位置
            top5_pool = top5_pool[torch.isin(top5_pool, valid_pos)]  # 有效长度不足5时只保留有效位置
            pos1 = top5_pool[torch.randint(0, top5_pool.numel(), (1,), device=x.device)]  # 槽1: top5内随机1个
            rest = valid_pos[~torch.isin(valid_pos, pos1)]  # 槽2候选: 其余有效位置
            if rest.numel() == 0:
                rest = valid_pos
            pos2 = rest[torch.randint(0, rest.numel(), (1,), device=x.device)]  # 槽2: 随机1个
            mask_pos = torch.cat((pos1, pos2), dim=0)

            mask_bool = torch.rand(x.shape[1]) > 1  # 生成随机的遮蔽布尔值
            mask_bool[mask_pos] = True  # 设置需要遮蔽的位置为True
            mask_bool = mask_bool.to(x.device)
            mask_bool_list.append(mask_bool.unsqueeze(0))

            mask_token = x[p, :][mask_pos]  # 获取原始x中对应位置的token

            n_pad = self.max_pred - 2
            mask_pos = torch.cat((mask_pos, torch.zeros(n_pad).long().to(x.device)), dim=0)
            mask_token = torch.cat((mask_token, torch.zeros(n_pad).long().to(x.device)), dim=0)

            mask_token_list.append(mask_token.unsqueeze(0))
            mask_pos_list.append(mask_pos.unsqueeze(0))

        mask_bool = torch.cat(mask_bool_list, dim=0)
        mask_x = torch.masked_fill(x, mask_bool, 1)

        mask_token_list = torch.cat(mask_token_list, dim=0)
        mask_pos_list = torch.cat(mask_pos_list, dim=0)
        mask_token_list = mask_token_list.long()
        mask_pos_list = mask_pos_list.long()

        return mask_x, mask_token_list, mask_pos_list


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size):
        super().__init__(vocab_size, embed_size, padding_idx=0)


class BERTEmbedding(nn.Module):

    def __init__(self, vocab_size, embed_size, dropout, max_len):
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        # self.position = PositionalEmbedding(d_model=embed_size,max_len=max_len)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence):
        x = self.token(sequence)  # + self.position(sequence)
        return self.dropout(x)


class Attention(nn.Module):

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):

    def __init__(self, h, d_model, dropout):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


class SublayerConnection(nn.Module):

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class TransformerBlock(nn.Module):

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)


def get_attn_pad_mask(x):
    atten_mask1 = (x == 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
    atten_mask2 = atten_mask1.transpose(-2, -1)
    atten_mask = torch.add(atten_mask1, atten_mask2)
    return atten_mask


class MSBERT(nn.Module):

    def __init__(self, vocab_size, hidden, n_layers, attn_heads, dropout, max_len, max_pred, mask_cls=Mask2):

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.feed_forward_hidden = hidden * 4
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden, dropout=dropout, max_len=max_len)

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])
        self.mask = Mask(max_pred)
        self.mask2 = mask_cls(max_pred)
        self.fc2 = nn.Linear(hidden, vocab_size, bias=False)
        self.activ2 = nn.GELU()
        self.linear = nn.Linear(hidden, hidden)
        self.intensity_linear = nn.Linear(1, hidden)
        # self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, input_id, intensity):
        mask_x1, mask_token1, mask_pos1 = self.mask2(input_id, intensity)
        mask_x2, mask_token2, mask_pos2 = self.mask2(input_id, intensity)

        inten = intensity.transpose(1, 2)
        inten = self.intensity_linear(inten)

        output1 = self.embedding(mask_x1)
        output1 = output1 + inten
        output2 = self.embedding(mask_x2)
        output2 = output2 + inten

        atten_mask1 = (mask_x1 == 0).unsqueeze(1).repeat(1, mask_x1.size(1), 1).unsqueeze(1)
        for transformer in self.transformer_blocks:
            output1 = transformer.forward(output1, atten_mask1)
        pool1 = torch.matmul(intensity, output1)
        pool1 = pool1 / intensity.shape[1]

        atten_mask2 = (mask_x2 == 0).unsqueeze(1).repeat(1, mask_x2.size(1), 1).unsqueeze(1)
        for transformer in self.transformer_blocks:
            output2 = transformer.forward(output2, atten_mask2)
        pool2 = torch.matmul(intensity, output2)
        pool2 = pool2 / intensity.shape[1]

        mask_pos1 = mask_pos1[:, :, None].expand(-1, -1, self.hidden)
        h_masked1 = torch.gather(output1, 1, mask_pos1)
        h_masked1 = self.activ2(self.linear(h_masked1))
        logits_lm1 = self.fc2(h_masked1)

        mask_pos2 = mask_pos2[:, :, None].expand(-1, -1, self.hidden)
        h_masked2 = torch.gather(output2, 1, mask_pos2)
        h_masked2 = self.activ2(self.linear(h_masked2))
        logits_lm2 = self.fc2(h_masked2)

        return logits_lm1, mask_token1, pool1, logits_lm2, mask_token2, pool2

    def predict(self, input_id, intensity):

        output = self.embedding(input_id)
        inten = intensity.transpose(1, 2)
        inten = self.intensity_linear(inten)
        output = output + inten

        atten_mask = (input_id == 0).unsqueeze(1).repeat(1, input_id.size(1), 1).unsqueeze(1)
        for idx, transformer in enumerate(self.transformer_blocks):
            output = transformer.forward(output, atten_mask)
        pool = torch.matmul(intensity, output)
        pool = pool / intensity.shape[1]
        return pool


class MSBERT_MaskTop5PlusRandom(MSBERT):
    """掩码变体: 强度 top5 的有效峰中掩 1 个 + 其余有效峰随机掩 1 个"""

    def __init__(self, vocab_size, hidden, n_layers, attn_heads, dropout, max_len, max_pred):
        super().__init__(vocab_size, hidden, n_layers, attn_heads, dropout, max_len, max_pred,
                         MaskTop5PlusRandom)














