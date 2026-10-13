"""Clustering evaluation metrics."""

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import cluster, normalized_mutual_info_score


def accuracy_score(y_true, y_pred):
    """Return clustering accuracy after optimal label assignment."""
    contingency = cluster.contingency_matrix(y_true, y_pred)
    row_indices, column_indices = linear_sum_assignment(-contingency)
    return contingency[row_indices, column_indices].sum() / contingency.sum()


def purity_score(y_true, y_pred):
    """Return clustering purity."""
    contingency = cluster.contingency_matrix(y_true, y_pred)
    return np.amax(contingency, axis=0).sum() / contingency.sum()


def cluster_metric(y_true, y_pred):
    """Return clustering accuracy, normalized mutual information, and purity."""
    return (
        accuracy_score(y_true, y_pred),
        normalized_mutual_info_score(y_true, y_pred),
        purity_score(y_true, y_pred),
    )
