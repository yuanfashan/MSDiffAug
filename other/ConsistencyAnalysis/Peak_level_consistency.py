# -*- coding: utf-8 -*-
"""峰级一致性分析入口: 原谱 / 随机增广 / 扩散增广 / VAE 增广 对比。

加载训练集原谱, 用冻结的扩散模型与 VAE 生成增广视图, 统计保留率 (按强度分组)、
新峰率、m/z 位移分布、余弦相似度。
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

import torch

from trainner import Trainer, DataGenerator2
from utils.diff_utils import set_seed
from utils.diff_config import get_config


def main():
    args = get_config()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # 数据：训练集原谱作谱库，其生成谱 (diff/random 视图) 作 query
    data_generator = DataGenerator2()

    trainer = Trainer(args, device, data_generator)
    # 模型冻结：峰级一致性评估与 epoch 无关，只执行一次
    top1_accuracy, best_accuracy = trainer.test_step(
        0, 0.0, data_generator.train_data, data_generator.train_data,
        data_generator.smiles, data_generator.smiles)


if __name__ == "__main__":
    main()
