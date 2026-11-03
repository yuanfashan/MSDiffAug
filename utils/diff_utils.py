import math
import torch
import numpy as np
from scipy.sparse import csr_matrix
import random
import torch.nn as nn


def set_seed(seed):
    '''Fix all of random seed for reproducible training'''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # only add when conv in your model


def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    # print("actual", len(actual))
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    return sum_recall / true_users


def ndcg_k(actual, predicted, topk):
    res = 0
    for user_id in range(len(actual)):
        k = min(topk, len(actual[user_id]))
        idcg = idcg_k(k)
        dcg_k = sum([int(predicted[user_id][j] in
                         set(actual[user_id])) / math.log(j + 2, 2) for j in range(topk)])
        res += dcg_k / idcg
    return res / float(len(actual))


# Calculates the ideal discounted cumulative gain at k
def idcg_k(k):
    res = sum([1.0 / math.log(i + 2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res


def get_full_sort_score(epoch, answers, pred_list):
    recall, ndcg = [], []
    for k in [5, 10, 15, 20]:
        recall.append(recall_at_k(answers, pred_list, k))
        ndcg.append(ndcg_k(answers, pred_list, k))
    post_fix = {
        "Epoch": epoch,
        "HIT@5": '{:.4f}'.format(recall[0]), "NDCG@5": '{:.4f}'.format(ndcg[0]),
        "HIT@10": '{:.4f}'.format(recall[1]), "NDCG@10": '{:.4f}'.format(ndcg[1]),
        "HIT@20": '{:.4f}'.format(recall[3]), "NDCG@20": '{:.4f}'.format(ndcg[3])
    }
    print(post_fix)


class BPRLoss(nn.Module):
    """BPRLoss, based on Bayesian Personalized Ranking

    Args:
        - gamma(float): Small value to avoid division by zero

    Shape:
        - Pos_score: (N)
        - Neg_score: (N), same shape as the Pos_score
        - Output: scalar.

    Examples::

        >>> loss = BPRLoss()
        >>> pos_score = torch.randn(3, requires_grad=True)
        >>> neg_score = torch.randn(3, requires_grad=True)
        >>> output = loss(pos_score, neg_score)
        >>> output.backward()
    """

    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos_score, neg_score):
        loss = -torch.log(self.gamma + torch.sigmoid(pos_score - neg_score)).mean()
        return loss


def generate_rating_matrix_valid(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-2]:  #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix


def generate_rating_matrix_test(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-1]:  #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix


def neg_sample(item_set, item_size):
    item = random.randint(1, item_size - 1)
    while item in item_set:
        item = random.randint(1, item_size - 1)
    return item


def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather consts for $t$ and reshape to feature map shape"""
    c = consts.gather(-1, t.reshape(-1))
    # print("debug c", c.shape)
    return c.reshape(-1, t.shape[-1], 1)


def q_xt_x0(x0, t, alpha_bar):
    """get the noised x and noise"""
    # alpha_bar: (step_num)
    # print("x0", x0.shape)  # (B, L, h)
    # print("t", t.shape)  # (B, L)
    # print("alpha_bar", alpha_bar.shape)  # (1000)

    mean = gather(alpha_bar, t) ** 0.5 * x0  # (bs, max_len, hidden_size)
    # print("mean", mean.shape)
    var = 1 - gather(alpha_bar, t)  # (bs, 1, 1, 1)
    eps = torch.randn_like(x0).to(x0.device)
    return mean + (var ** 0.5) * eps, eps  # also returns noise


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    # print("beta", beta.shape)
    # print("t", t.shape)
    a = (1 - beta).cumprod(dim=0).index_select(0, t.reshape(-1) + 1).view(t.shape[0], t.shape[1], 1)
    # print("a", a.shape)
    return a


def p_xt(xt, noise, t, next_t, beta, eta):
    at = compute_alpha(beta, t.long())
    at_next = compute_alpha(beta, next_t.long())
    x0_t = (xt - noise * (1 - at).sqrt()) / at.sqrt()
    c1 = (eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt())
    c2 = ((1 - at_next) - c1 ** 2).sqrt()
    eps = torch.randn(xt.shape, device=xt.device)
    xt_next = at_next.sqrt() * x0_t + c1 * eps + c2 * noise
    return xt_next


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, checkpoint_path, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def compare(self, score):
        for i in range(len(score)):
            # 有一个指标增加了就认为是还在涨
            if score[i] > self.best_score[i] + self.delta:
                return False
        return True

    def __call__(self, score, model):
        # score HIT@10 NDCG@10

        if self.best_score is None:
            self.best_score = score
            self.score_min = np.array([0] * len(score))
            self.save_checkpoint(score, model)
        elif self.compare(score):
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            # ({self.score_min:.6f} --> {score:.6f}) # 这里如果是一个值的话输出才不会有问题
            print(f"Validation score increased.  Saving model ...")
        torch.save(model.state_dict(), self.checkpoint_path)
        self.score_min = score


class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def SearchTopATS_MemoryEfficien(dataset_arr, query_arr, dataset_smiles, query_smiles,
                                batch=500, dataset_chunk_size=5000, device='cuda',
                                fp_radius=2, fp_bits=2048,
                                topk_list=(500, 250, 100, 50, 25, 10)):
    """多档位 Top-k 类似物检索评估: ATS@k (平均最大指纹相似度)。

    对每个 query 嵌入, 在谱库 (dataset) 嵌入中分块计算余弦最近邻, 只保留
    top-K (K = max(topk_list)); 再对每个 k, 计算 query 真实结构与 top-k 命中
    结构的 Morgan 指纹 Tanimoto 相似度, 每个 query 取最大值, 对所有 query 取
    平均 (无效 SMILES / 无有效命中不计入)。

    返回列表, 顺序与降序排列的 topk_list 一致:
      [ATS@500, ATS@250, ATS@100, ATS@50, ATS@25, ATS@10]
    """
    from rdkit import Chem
    from rdkit import RDLogger
    from rdkit.Chem import DataStructs, rdFingerprintGenerator
    import torch.nn.functional as F
    RDLogger.DisableLog('rdApp.error')  # 数据中混有非 SMILES 编号, 屏蔽 RDKit 解析报错刷屏

    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=fp_radius, fpSize=fp_bits)

    if torch.cuda.is_available() and device == 'cuda':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    torch.cuda.empty_cache()

    topk_list = tuple(sorted(set(topk_list), reverse=True))
    K = topk_list[0]

    n_dataset = len(dataset_arr)
    n_queries = len(query_arr)
    if n_queries == 0:
        return [float("nan")] * len(topk_list)

    best_scores = torch.full((n_queries, K), float("-inf"), device=device)
    best_indices = torch.full((n_queries, K), -1, dtype=torch.int64, device=device)

    with torch.no_grad():
        for dataset_start in range(0, n_dataset, dataset_chunk_size):
            dataset_end = min(dataset_start + dataset_chunk_size, n_dataset)
            dataset_chunk = F.normalize(
                torch.from_numpy(dataset_arr[dataset_start:dataset_end]).float().to(device),
                dim=1)

            for query_start in range(0, n_queries, batch):
                query_end = min(query_start + batch, n_queries)
                query_batch = F.normalize(
                    torch.from_numpy(query_arr[query_start:query_end]).float().to(device),
                    dim=1)
                scores = torch.mm(query_batch, dataset_chunk.t())  # (B, chunk)

                q_scores = best_scores[query_start:query_end]
                q_indices = best_indices[query_start:query_end]
                chunk_indices = torch.arange(dataset_start, dataset_end, device=device)
                chunk_indices = chunk_indices.unsqueeze(0).expand(scores.shape[0], -1)

                all_scores = torch.cat([q_scores, scores], dim=1)  # (B, K+chunk)
                all_indices = torch.cat([q_indices, chunk_indices], dim=1)
                top_vals, top_pos = torch.topk(all_scores, k=K, dim=1)
                best_scores[query_start:query_end] = top_vals
                best_indices[query_start:query_end] = all_indices.gather(1, top_pos)

            del dataset_chunk
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    best_indices = best_indices.cpu().numpy()

    fp_cache = {}

    def get_fp(smi):
        if not smi:
            return None
        if smi not in fp_cache:
            mol = Chem.MolFromSmiles(smi)
            fp_cache[smi] = fp_gen.GetFingerprint(mol) if mol is not None else None
        return fp_cache[smi]

    per_query_max = [[] for _ in topk_list]
    for i in range(n_queries):
        fp_q = get_fp(query_smiles[i])
        if fp_q is None:
            continue
        valid = best_indices[i][best_indices[i] >= 0]
        hit_fps = [get_fp(dataset_smiles[idx]) for idx in valid]
        hit_fps = [fp for fp in hit_fps if fp is not None]
        if not hit_fps:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp_q, hit_fps)
        for j, k in enumerate(topk_list):
            per_query_max[j].append(max(sims[:k]))

    return [float(np.mean(v)) if v else float("nan") for v in per_query_max]