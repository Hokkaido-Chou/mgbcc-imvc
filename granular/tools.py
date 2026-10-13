"""Utilities for cross-view granular-ball relations."""

import numpy as np
import torch

from granular.base import GBList


def relation_of_views_gblists(view0: GBList, view1: GBList, threshold=0.1):
    """Build a relation when two balls share enough original sample IDs."""
    relation = np.zeros((len(view0), len(view1)), dtype=np.float32)
    right_sets = [set(ball.indices) for ball in view1.granular_balls]
    for i, left_ball in enumerate(view0.granular_balls):
        left_ids = set(left_ball.indices)
        for j, right_ids in enumerate(right_sets):
            overlap = len(left_ids & right_ids)
            if overlap / len(left_ids) > threshold or overlap / len(right_ids) > threshold:
                relation[i, j] = 1.0
    return torch.from_numpy(relation).to(view0.data.device)


def merge_tensors(n, m, tensor1, tensor2, tensor3, tensor4):
    """Combine four relation blocks into one square matrix."""
    merged = torch.zeros((n + m, n + m), device=tensor1.device)
    merged[:n, :n] = tensor1
    merged[:n, n:] = tensor2
    merged[n:, :n] = tensor3
    merged[n:, n:] = tensor4
    return merged
