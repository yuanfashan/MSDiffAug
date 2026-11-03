import argparse
from typing import List


def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ('true', '1', 'yes', 'on')


def get_config():
    parser = argparse.ArgumentParser()
    # ************SASRec*******************
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu')
    parser.add_argument("--dataset", default="Beauty",
                        choices=['ml-1m', 'Beauty', 'Sports_and_Outdoors', 'Yelp', 'Toys_and_Games'],
                        help="Choose the dataset")
    # parser.add_argument("--model_name", default="diffsas", help="Choose the model")
    # --model_name 默认是 diffsas，跑不同变体时要记得传对应的名称（如 --model_name MSMaskPredict、--model_name ContraDiff），否则各变体的模型会互相覆盖。

    parser.add_argument("--diffusion_epoch", type=int, default=100,
                        help='epoch of pretrained diffusion checkpoints to load '
                             '(diff_augment_epoch_X.pt / spaced_diffusion_epoch_X.pt)')  ####检查点开始加载位置，无初始化权权重
    parser.add_argument("--resume_checkpoint", default="latest")######从最新开始加载，有初始化权重#latest, none
    parser.add_argument("--check_path", default='', type=str, help="the save path of checkpoints for different running")
    parser.add_argument('--epochs', type=int, default=400, help='Number of training epochs')
    parser.add_argument('--warm_up_epochs', type=int, default=-1, help='Number of warmup epochs')
    parser.add_argument('--filter_num', type=int, default=5, help='filter_num')
    parser.add_argument('--train_batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=512, help='Batch size for testing')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')

    parser.add_argument('--n_layers', type=int, default=2, help='Number of encoder layers')
    parser.add_argument('--n_heads', type=int, default=2, help='Number of attention heads')
    parser.add_argument('--hidden_size', type=int, default=64, help='Hidden size (embedding size)')
    parser.add_argument('--inner_size', type=int, default=128, help='Dimensionality in feed-forward layer')
    parser.add_argument('--sasrec_dropout_prob', type=float, default=0.5, help='Hidden dropout probability of SASRec')
    parser.add_argument('--attn_dropout_prob', type=float, default=0.5, help='Hidden dropout probability of SASRec')
    parser.add_argument('--hidden_act', type=str, default='gelu', help='Activation function for hidden layers')
    parser.add_argument('--layer_norm_eps', type=float, default=1e-12, help='Epsilon value for layer normalization')

    parser.add_argument('--initializer_range', type=float, default=0.02, help='Initializer range for model parameters')
    parser.add_argument('--loss_type', type=str, default='BPR', help='Type of loss function')
    parser.add_argument('--max_seq_length', type=int, default=50, help='max sequence length')

    parser.add_argument('--temperature', type=float, default=1.0, help='temperature for nce loss')
    parser.add_argument('--seed', type=int, default=2024, help='random seed for diffusion training')#扩散模型的随机种子
    parser.add_argument('--seed_ms', type=int, default=2025, help='random seed for msencoder training')#表征学习模型的随机种子

    # #*************Diffusion***************
    parser.add_argument("--num_hidden_layers", type=int, default=2, help='num_hidden_layers for the bert encoder.')
    parser.add_argument("--intermediate_size", type=int, default=128, help='intermediate_size.')
    parser.add_argument("--num_attention_heads", type=int, default=2, help='num_attention_heads.')
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.2, help='attention_probs_dropout_prob.')
    parser.add_argument("--max_position_embeddings", type=int, default=50, help='max_position_embeddings.')
    parser.add_argument("--max_relative_positions", type=int, default=-1, help='max_relative_positions.')
    parser.add_argument("--type_vocab_size", type=int, default=0, help='type_vocab_size.')
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.2, help='hidden_dropout_prob.')
    parser.add_argument("--is_decoder", type=str2bool, default=False, help='is decoder')
    parser.add_argument("--mask_type", type=str, default='prob', help='mask_type.')
    parser.add_argument("--pad_token_id", type=int, default=0, help='padding id.')
    parser.add_argument("--pos_att_type", type=List[str], default=["p2c", "c2p"], help='pos_att_type.')
    parser.add_argument("--relative_attention", type=str2bool, default=True, help='relative_attention')
    parser.add_argument("--position_biased_input", type=str2bool, default=False, help='position_biased_input')

    parser.add_argument('--alpha', type=float, default=0.5, help='ratio of constrastive learning loss')
    parser.add_argument('--beta', type=float, default=0.2, help='ratio of diffusion nll loss')
    parser.add_argument('--gamma', type=float, default=0.0, help='ratio of diffusion mse loss')
    parser.add_argument('--mlm_probability_train', type=float, default=0.1, help='mlm_probability for train')
    parser.add_argument('--mlm_probability', type=float, default=0.1, help='mlm_probability')

    # 扩散增强强度 (ablition7: (m/z, 强度) 联合预测)
    parser.add_argument("--save_dir", default="/home/ausu/projects/yfs/MSDiffAugServer/ablition8/p",
                        help='save dir for diffusion/msencoder checkpoints')#扩散模型保存地址
    parser.add_argument('--mask_ratio', type=float, default=0.1,
                        help='ratio of peaks corrupted by diffusion augmentation')#扩散比例ρ
    parser.add_argument('--intensity_loss_weight', type=float, default=100.0,
                        help='ratio of intensity regression loss in joint (m/z, intensity) diffusion prediction')#强度回归损失权重λ
    parser.add_argument('--t_max', type=int, default=100,
                        help='max diffusion timestep sampled during training (limits corruption level)')#扩散最大采样时间步
    parser.add_argument('--ddim_sampling_steps', type=int, default=10,
                        help='number of DDIM reverse-chain steps for iterative augmented view generation')#增强视图迭代反向链采样步数
    parser.add_argument("--noise_schedule", default="cosine")#扩散调度函数
    parser.add_argument("--beta_start", type=float, default=1e-5)  # β_start
    parser.add_argument("--beta_end", type=float, default=1e-3,
                        help='显式 beta 调度终点 (如 1e-3)')  # β_end
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--microbatch", type=int, default=0)
    parser.add_argument("--ema_rate", type=float, default=1.0)
    parser.add_argument("--schedule_sampler", default="lossaware")
    parser.add_argument("--diffusion_steps", type=int, default=1000)#时间步
    parser.add_argument("--inference_sampling_steps", type=int, default=5)
    parser.add_argument("--use_plm_init", default="no")
    parser.add_argument("--notes", default="folder-notes")
    parser.add_argument("--seq_len", type=int, default=100)#序列长度
    parser.add_argument("--dropout", type=float, default=0.15)  # 扩散的dropout
    parser.add_argument("--use_fp16", type=str2bool, default=True)  # 采用AMP混合精度训练加快训练速度
    parser.add_argument("--fp16_scale_growth", type=float, default=0.001)
    parser.add_argument("--gradient_clipping", type=float, default=-1.0)
    parser.add_argument("--learn_sigma", type=str2bool, default=False)
    parser.add_argument("--use_kl", type=str2bool, default=False)
    parser.add_argument("--predict_xstart", type=str2bool, default=True)

    # 训练
    # parser.add_argument("--rescale_timesteps", type=str2bool, default=True)
    # parser.add_argument("--timestep_respacing", default="")
    # 使用
    parser.add_argument("--rescale_timesteps", type=str2bool, default=False)  # 改为 False，让梯度更陡峭
    parser.add_argument("--timestep_respacing", default="100")

    parser.add_argument("--rescale_learned_sigmas", type=str2bool, default=False)
    parser.add_argument("--sigma_small", type=str2bool, default=False)
    parser.add_argument("--emb_scale_factor", type=float, default=1.0)

    return parser.parse_args()