# -*- coding: utf-8 -*-
"""化学合理性分析入口: 原谱对照 (original) / 扩散增广 (diff) / 随机偏移 (random) / VAE 增广 (vae)。

加载训练集原谱, 用冻结的扩散模型与冻结 VAE 生成增广视图, 在七项化学约束上评估:
C1 前体质量约束 / C2 谱内一致性 / C3 前体中性丢失 / C4 位移即中性丢失 /
C5 谱内中性丢失 / C6 谱内组合中性丢失 / C7 碎片中性丢失。
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

    # 数据：训练集原谱作增广来源，其生成谱 (diff/random/vae 视图) 作评估对象
    data_generator = DataGenerator2()

    trainer = Trainer(args, device, data_generator)
    # 模型冻结：化学合理性评估与 epoch 无关，只执行一次
    summary = trainer.evaluate_chemical_plausibility(data_generator.train_data)


if __name__ == "__main__":
    main()
