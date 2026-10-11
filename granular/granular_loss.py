"""Paper-path granular-ball contrastive losses.

This module exposes only the two loss classes used by the ICGBC manuscript and
its seven retained experiments:

- :class:`MultiviewGCLoss` — the V1 p=2 standard cross-view overlap loss.
- :class:`PermutedRelationMultiviewGCLoss` — the density-matched negative
  control used by ``experiments/overlap_relation_ablation/``.

All anchor-, reliability-, topology-, gating-, and selector-style losses
developed during the research phase are quarantined under
``granular._exploration.losses_exploration``; they are intentionally not
imported from this module.
"""
import numpy as np
import torch
from granular.base import GBList, MVGBList
from granular.tools import merge_tensors, relation_of_views_gblists


class MultiviewGCLoss(torch.nn.Module):
    def __init__(self, temperature=1.):
        super(MultiviewGCLoss, self).__init__()
        self.t = temperature

    def forward(self, views: MVGBList):
        # 统一设备
        device = views[0].data.device
        loss = torch.tensor(0., device=device)
        # 两两视图之间进行对比
        num_views = len(views)
        for i in range(num_views):
            # mask_i_intra = views[i].affinity()
            mask_i_intra = torch.eye(len(views[i]), device=device)
            for j in range(i + 1, num_views):
                # 计算掩码
                # mask_j_intra = views[j].affinity()
                mask_j_intra = torch.eye(len(views[j]), device=device)
                mask_inter = relation_of_views_gblists(views[i], views[j])
                # 两个视图的粒球数量
                ni, nj = len(views[i]), len(views[j])
                # 合并视图内和视图间的掩码矩阵
                pos_mask = merge_tensors(ni, nj, mask_i_intra, mask_inter, mask_inter.T, mask_j_intra).to(device)
                # neg_mask = 1 - pos_mask
                neg_mask = torch.ones_like(pos_mask).to(device) - pos_mask
                num_ins = ni + nj
                # idx = torch.arange(0, num_ins)
                # 修正正样本对掩码
                # pos_mask[idx, idx] = 0
                centers_i = views[i].get_centers()
                centers_j = views[j].get_centers()
                x = torch.cat((centers_i, centers_j), dim=0)
                # 计算相似度，这里就是矩阵相乘
                norm_x = torch.norm(x, p=2, dim=1, keepdim=True)
                sim_x = x @ x.T / (norm_x @ norm_x.T + 1e-12)
                # 考虑用cross entropy 重写
                sim_pos = pos_mask * sim_x / self.t
                sim_neg = neg_mask * sim_x / self.t
                exp_sim_neg = torch.sum(torch.exp(sim_neg), dim=1, keepdim=True).expand((num_ins, num_ins))
                expsum_sim = torch.exp(sim_pos) + exp_sim_neg
                # expsum_sim = exp_sim_neg
                if pos_mask.sum().item() > 0:
                    loss += torch.sum(-(sim_pos - torch.log(expsum_sim) * pos_mask)) / pos_mask.sum()
        return loss / (num_views * (num_views - 1) / 2)


class PermutedRelationMultiviewGCLoss(torch.nn.Module):
    """Density-matched negative control for the standard overlap relation.

    The standard binary relation is constructed first, then its target-ball
    columns are permuted with a deterministic Sattolo cycle.  This preserves
    the exact number of cross-view positives and both endpoint-degree
    distributions, while breaking which target ball each source ball is
    associated with.  The standard V1 loss is intentionally left untouched.
    """

    def __init__(self, temperature=1.0, relation_seed=0):
        super().__init__()
        self.t = temperature
        self.relation_seed = int(relation_seed)
        self.last_pair_stats = None

    @staticmethod
    def _sattolo_permutation(size, seed, device):
        if size <= 1:
            return torch.arange(size, device=device)
        permutation = np.arange(size, dtype=np.int64)
        rng = np.random.default_rng(seed)
        for index in range(size - 1, 0, -1):
            swap_index = int(rng.integers(0, index))
            permutation[index], permutation[swap_index] = permutation[swap_index], permutation[index]
        return torch.as_tensor(permutation, device=device, dtype=torch.long)

    def forward(self, views: MVGBList):
        device = views[0].data.device
        loss = torch.tensor(0.0, device=device)
        num_views = len(views)
        pair_stats = []
        for i in range(num_views):
            mask_i_intra = torch.eye(len(views[i]), device=device)
            for j in range(i + 1, num_views):
                mask_j_intra = torch.eye(len(views[j]), device=device)
                base_relation = relation_of_views_gblists(views[i], views[j])
                ni, nj = len(views[i]), len(views[j])
                pair_seed = self.relation_seed + 1000003 * i + 10007 * j + 101 * ni + nj
                permutation = self._sattolo_permutation(nj, pair_seed, device)
                mask_inter = base_relation[:, permutation]
                base_edges = int(base_relation.sum().item())
                retained_edges = int((base_relation * mask_inter).sum().item())
                pair_stats.append({
                    "view_i": i,
                    "view_j": j,
                    "num_balls_i": ni,
                    "num_balls_j": nj,
                    "base_edges": base_edges,
                    "control_edges": int(mask_inter.sum().item()),
                    "retained_edge_fraction": retained_edges / max(base_edges, 1),
                    "fixed_points": int((permutation == torch.arange(nj, device=device)).sum().item()),
                })
                pos_mask = merge_tensors(
                    ni, nj, mask_i_intra, mask_inter, mask_inter.T, mask_j_intra
                ).to(device)
                neg_mask = torch.ones_like(pos_mask) - pos_mask
                num_ins = ni + nj
                x = torch.cat((views[i].get_centers(), views[j].get_centers()), dim=0)
                norm_x = torch.norm(x, p=2, dim=1, keepdim=True)
                sim_x = x @ x.T / (norm_x @ norm_x.T + 1e-12)
                sim_pos = pos_mask * sim_x / self.t
                sim_neg = neg_mask * sim_x / self.t
                exp_sim_neg = torch.sum(torch.exp(sim_neg), dim=1, keepdim=True).expand((num_ins, num_ins))
                expsum_sim = torch.exp(sim_pos) + exp_sim_neg
                if pos_mask.sum().item() > 0:
                    loss += torch.sum(-(sim_pos - torch.log(expsum_sim) * pos_mask)) / pos_mask.sum()
        self.last_pair_stats = {
            "mode": "permuted_relation",
            "relation_seed": self.relation_seed,
            "pairs": pair_stats,
            "base_edges": sum(item["base_edges"] for item in pair_stats),
            "control_edges": sum(item["control_edges"] for item in pair_stats),
            "retained_edge_fraction_mean": float(np.mean([item["retained_edge_fraction"] for item in pair_stats])) if pair_stats else 0.0,
            "fixed_points": sum(item["fixed_points"] for item in pair_stats),
        }
        return loss / (num_views * (num_views - 1) / 2)
