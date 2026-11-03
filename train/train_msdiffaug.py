import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
for p in (_SCRIPT_DIR, _REPO_ROOT):
    if p not in sys.path:
        sys.path.append(p)

from model.representation.trainner import Trainer
from utils.diff_utils import set_seed
from utils.diff_config import get_config
from utils.ms_utils import ParseOrbitrap, ParseOrbitrap_ik, ModelEmbed, SearchTop_MemoryEfficient
from data_process.process_data import MakeTrainData
import random
import numpy as np
from torch.utils.data import Dataset
import math
from functools import partial


def _seed_worker(worker_id, base_seed):
    np.random.seed(base_seed + worker_id)


class MyDataSet(Dataset):
    def __init__(self, input_ids, intensity, mask_ratio=0.1):
        '''
        初始化数据集，接收两个输入参数：input_ids 和 intensity

        :param input_ids: 输入 ID，形状为 (样本数, 100)
        :param intensity: 强度值，形状为 (样本数, 100)
        :param mask_ratio: 用于随机掩蔽的位置比例（默认为 0.1）
        '''
        self.input_ids = input_ids  # 记录 input_ids
        self.intensity = intensity  # 记录 intensity
        self.mask_ratio = mask_ratio  # 掩蔽的比例

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
        # 获取当前样本的 input_ids 和 intensity (view, collate 时拷贝)
        input_ids = self.input_ids[idx]
        intensity = self.intensity[idx]

        # 生成 attention_mask，input_id 为 0 时 mask 为 0.0，其他为 1.0
        attention_mask = (input_ids != 0).float()

        # 生成 masked_indices0，按 0.1 的比率随机标记非零位置为 1
        # 获取所有非零位置的索引
        non_zero_indices = (input_ids != 0).nonzero(as_tuple=True)[0]

        # 随机选择 10% 的非零位置
        # num_masked = int(len(non_zero_indices) * self.mask_ratio)
        # num_masked = 2
        # num_masked = max(int(len(non_zero_indices) * self.mask_ratio), 1)
        x = max(int(len(non_zero_indices) * self.mask_ratio), 1)
        num_masked = max(int(round(5 * math.log(x) / math.log(10))), 1)
        masked_indices = np.random.choice(non_zero_indices.numpy(), num_masked, replace=False)

        # 创建 masked_indices0，初始全为 0
        masked_indices0 = torch.zeros_like(input_ids, dtype=torch.bool)

        # 将随机选择的位置标记为 1
        masked_indices0[masked_indices] = 1

        # 第二个视图的腐蚀掩码: 独立随机再选一组 15% 非零位置
        masked_indices1_idx = np.random.choice(non_zero_indices.numpy(), num_masked, replace=False)
        masked_indices1 = torch.zeros_like(input_ids, dtype=torch.bool)
        masked_indices1[masked_indices1_idx] = 1

        # 返回包含 input_ids, intensity, attention_mask 和 masked_indices0 的字典
        return {
            'input_ids': input_ids,
            'intensity': intensity,
            'attention_mask': attention_mask,
            'masked_indices0': masked_indices0,
            'masked_indices1': masked_indices1
        }


def mask_input_ids(self, input_ids, p):
    mask_indices = []
    for token in input_ids:
        if token == 0:
            mask_indices.append(0)  # 元素为0的位置不参与mask
        else:
            mask_indices.append(1 if random.random() < p else 0)
    return mask_indices
import torch
from torch.utils.data import DataLoader

class DataGenerator2:
    def __init__(self, max_len=100, mask_ratio=0.1, seed=2025):
        # 超参数
        self.max_len = max_len

        # 加载数据
        train_ref, msms1, precursor1, smiles1 = ParseOrbitrap('/home/ausu/projects/yfs/data/GNPSdata_1/ob_train_ref.pickle')
        # train_ref, msms1, precursor1, smiles1 = ParseOrbitrap('/home/ausu/projects/yfs/data/train_data/1/ob_train_ref_1.pickle')  # 10%
        print(len(smiles1))
        train_ref, word2idx = MakeTrainData(msms1, precursor1, self.max_len)

        # 词汇表大小
        self.vocab_size = len(word2idx)
        self.word2idx = word2idx

        # 准备数据集: 堆叠成单一大 tensor, 避免每样本小 tensor 的列表开销
        input_ids, intensity, _, _ = zip(*train_ref)
        input_ids = torch.stack([torch.LongTensor(i) for i in input_ids])   # [N, 100]
        intensity = torch.stack([torch.FloatTensor(i) for i in intensity])  # [N, 1, 100]; cat 会把 1 维塌缩成 [N,100]

        # 创建数据集
        self.dataset = MyDataSet(input_ids, intensity, mask_ratio)

        # 生成数据加载器
        init_fn = partial(_seed_worker, base_seed=seed)
        loader_kwargs = dict(batch_size=64, num_workers=4, pin_memory=True,
                             persistent_workers=True, prefetch_factor=4,
                             worker_init_fn=init_fn)
        self.train_dataloader = DataLoader(self.dataset, shuffle=True, **loader_kwargs)
        self.valid_dataloader = DataLoader(self.dataset, shuffle=False, **loader_kwargs)  # 可以根据需要修改
        self.test_dataloader = DataLoader(self.dataset, shuffle=False, **loader_kwargs)  # 可以根据需要修改

        # 数据集的大小
        self.item_size = len(self.dataset)

    def get_dataloaders(self):
        return self.train_dataloader, self.valid_dataloader, self.test_dataloader




def main():
    args = get_config()
    set_seed(args.seed_ms)
    max_len = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    #训练数据
    data_generator = DataGenerator2(mask_ratio=args.mask_ratio)
    # data_generator = DataGenerator2(mask_ratio=0.20, seed=args.seed_ms)

    # 测试数据
    test_msms1, test_precursor1, test_smiles1, test_ik1 = ParseOrbitrap_ik('/home/ausu/projects/yfs/data/GNPSdata_1/ob_test_ref.pickle')
    test_msms2, test_precursor2, test_smiles2, test_ik2 = ParseOrbitrap_ik('/home/ausu/projects/yfs/data/GNPSdata_1/ob_test_query.pickle')
    test_ref, _ = MakeTrainData(test_msms1, test_precursor1, max_len)
    test_query, _ = MakeTrainData(test_msms2, test_precursor2, max_len)

    trainer = Trainer(args, device, data_generator)
    # trainer.train()
    epochs = 200
    train_time = []
    best_accuracy = 0.0  # 记录最佳准确率
    for epoch in range(epochs):
        t = trainer._train_one_epoch(epoch,  only_bert_train=True, is_diffusion_train=False)
        train_time.append(t)
        top1_accuracy, best_accuracy = trainer.test_step(epoch, best_accuracy, test_ref, test_query,
                                                         test_smiles1, test_smiles2, test_ik1, test_ik2)
        # top1_accuracy = trainer.test_step(train_ref, train_query, smiles1, smiles2)




if __name__ == "__main__":
    main()






