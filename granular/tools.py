import torch
from torch.nn.functional import one_hot
import numpy as np
from granular.base import GranularBall, GBList, contain_same_sample


# 两个视图的粒球集之间的关系
def relation_of_views_gblists(view0: GBList, view1: GBList, t=0.1):
    # 不同视图之间的实例具有对应关系，因此需要建立跨视图粒球之间的联系
    # 一种简单的思路是：直接按照实例对应关系建立联系，为此，需要知道每个粒球中包含哪些样本（索引）
    # 这个关系可以作为对比学习过程的掩码
    n0, n1 = len(view0), len(view1)
    mask = np.zeros((n0, n1), dtype=np.float32)
    for i in range(n0):
        set0 = set(view0[i].indices)
        for j in range(n1):
            # if contain_same_sample(view0[i], view1[j]):
            #     mask[i, j] = 1
            set1 = set(view1[j].indices)
            sub_set = set0 & set1
            if len(sub_set) / len(set0) > t or len(sub_set) / len(set1) > t:
                mask[i, j] = 1
    return torch.from_numpy(mask).to(view0.data.device)

def relation_and_overlap_of_views_gblists(view0: GBList, view1: GBList, t=0.1):
    """Return the binary relation mask and continuous overlap evidence."""
    n0, n1 = len(view0), len(view1)
    relation = np.zeros((n0, n1), dtype=np.float32)
    overlap = np.zeros((n0, n1), dtype=np.float32)
    sets0 = [set(ball.indices) for ball in view0.granular_balls]
    sets1 = [set(ball.indices) for ball in view1.granular_balls]
    for i, ids0 in enumerate(sets0):
        size0 = max(len(ids0), 1)
        for j, ids1 in enumerate(sets1):
            inter = len(ids0 & ids1)
            if inter == 0:
                continue
            score = max(inter / size0, inter / max(len(ids1), 1))
            overlap[i, j] = score
            if score > t:
                relation[i, j] = 1.0
    device = view0.data.device
    return torch.from_numpy(relation).to(device), torch.from_numpy(overlap).to(device)

def relation_of_views_gblists_tensor(view0: GBList, view1: GBList, t=0.1):
    y_parts0 = view0.y_parts
    y_parts1 = view1.y_parts
    num_gb = len(view0)
    # n * k
    one_hot0 = one_hot(y_parts0, num_classes=num_gb).float()
    # n * k
    one_hot1 = one_hot(y_parts1, num_classes=num_gb).float()
    mask = one_hot0.T @ one_hot1
    num_gb_set0 = one_hot0.sum(dim=0).view((-1, 1))
    num_gb_set1 = one_hot1.sum(dim=0).view((1, -1))
    num_gb_min = torch.min(num_gb_set0, num_gb_set1)
    mask =  (mask / num_gb_min) > t
    return mask.float()



def merge_tensors(n, m, tensor1, tensor2, tensor3, tensor4):
    # 创建一个大小为 (n+m) * (n+m) 的零张量
    merged_tensor = torch.zeros((n + m, n + m), device=tensor1.device)

    # 填充第一个 tensor 张量
    merged_tensor[:n, :n] = tensor1

    # 填充第二个 tensor 张量
    merged_tensor[:n, n:n + m] = tensor2

    # 填充第三个 tensor 张量
    merged_tensor[n:n + m, :n] = tensor3

    # 填充第四个 tensor 张量
    merged_tensor[n:n + m, n:n + m] = tensor4

    return merged_tensor


