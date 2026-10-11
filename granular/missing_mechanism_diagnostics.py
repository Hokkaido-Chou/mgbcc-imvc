"""Offline diagnostics for missing-view evidence and representation fusion.

This module is opt-in. Labels are used only after training for interpretation;
none of its outputs affect optimization, ball construction, or model selection.
"""

from itertools import combinations

import numpy as np
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from utils.score import cluster_metric


def _summary(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "min": 0.0, "q25": 0.0, "median": 0.0, "q75": 0.0, "max": 0.0, "mean": 0.0}
    quantiles = np.quantile(values, [0.0, 0.25, 0.5, 0.75, 1.0])
    return {
        "count": int(values.size),
        "min": float(quantiles[0]),
        "q25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "q75": float(quantiles[3]),
        "max": float(quantiles[4]),
        "mean": float(values.mean()),
    }


def _ball_record(ball):
    labels = ball.labels.detach().cpu().numpy().astype(np.int64).reshape(-1)
    if labels.size == 0:
        return {
            "ids": set(),
            "support": 0,
            "radius": None,
            "purity": None,
            "majority_label": None,
        }
    unique, counts = np.unique(labels, return_counts=True)
    majority_index = int(np.argmax(counts))
    return {
        "ids": set(map(int, ball.indices)),
        "support": int(ball.num_smp),
        "radius": float(ball.r.detach().cpu().item()),
        "purity": float(counts[majority_index] / labels.size),
        "majority_label": int(unique[majority_index]),
    }


def _largest_bipartite_component_fraction(mask):
    n0, n1 = mask.shape
    total = n0 + n1
    if total == 0:
        return 0.0
    adjacency = [[] for _ in range(total)]
    for i, j in np.argwhere(mask):
        adjacency[int(i)].append(n0 + int(j))
        adjacency[n0 + int(j)].append(int(i))
    seen = set()
    largest = 0
    for start in range(total):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            for neighbor in adjacency[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        largest = max(largest, size)
    return float(largest / total)


def _safe_spearman(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return {"n": int(x.size), "rho": None, "pvalue": None}
    value = spearmanr(x, y)
    rho = getattr(value, "statistic", getattr(value, "correlation", value[0]))
    pvalue = getattr(value, "pvalue", value[1])
    return {"n": int(x.size), "rho": float(rho), "pvalue": float(pvalue)}


def _cluster_metrics(features, labels, n_clusters, seed):
    prediction = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(features)
    acc, nmi, pur = cluster_metric(labels, prediction)
    return {
        "ACC": float(acc * 100.0),
        "NMI": float(nmi * 100.0),
        "ARI": float(adjusted_rand_score(labels, prediction) * 100.0),
        "PUR": float(pur * 100.0),
    }


def summarize_ball_and_relation_evidence(mv_gblist, threshold=0.1):
    view_records = []
    for view in mv_gblist.gblists:
        balls = [_ball_record(ball) for ball in view.granular_balls]
        view_records.append((view, balls))

    view_stats = []
    for view, balls in view_records:
        observed = set().union(*(ball["ids"] for ball in balls)) if balls else set()
        valid_balls = [ball for ball in balls if ball["support"] > 0]
        view_stats.append({
            "view": int(getattr(view, "source_view_index", -1)),
            "observed_count": int(len(observed)),
            "ball_count": int(len(balls)),
            "empty_ball_count": int(len(balls) - len(valid_balls)),
            "support": _summary([ball["support"] for ball in valid_balls]),
            "radius": _summary([ball["radius"] for ball in valid_balls]),
            "purity_diagnostic_only": _summary([ball["purity"] for ball in valid_balls]),
            "mixed_ball_fraction_diagnostic_only": float(np.mean([ball["purity"] < 1.0 for ball in valid_balls])) if valid_balls else 0.0,
        })

    pair_stats = []
    for left_index, right_index in combinations(range(len(view_records)), 2):
        left_view, left = view_records[left_index]
        right_view, right = view_records[right_index]
        left_observed = set().union(*(ball["ids"] for ball in left)) if left else set()
        right_observed = set().union(*(ball["ids"] for ball in right)) if right else set()
        relation = np.zeros((len(left), len(right)), dtype=bool)
        edge_support = []
        edge_min_purity = []
        edge_max_radius = []
        edge_correct = []
        edge_overlap_score = []
        for i, ball_left in enumerate(left):
            for j, ball_right in enumerate(right):
                if ball_left["support"] <= 0 or ball_right["support"] <= 0:
                    continue
                intersection = len(ball_left["ids"] & ball_right["ids"])
                if intersection == 0:
                    continue
                score = max(intersection / ball_left["support"], intersection / ball_right["support"])
                if score <= threshold:
                    continue
                relation[i, j] = True
                edge_support.append(intersection)
                edge_min_purity.append(min(ball_left["purity"], ball_right["purity"]))
                edge_max_radius.append(max(ball_left["radius"], ball_right["radius"]))
                edge_correct.append(float(ball_left["majority_label"] == ball_right["majority_label"]))
                edge_overlap_score.append(score)
        row_degree = relation.sum(axis=1) if relation.shape[0] else np.asarray([], dtype=np.int64)
        col_degree = relation.sum(axis=0) if relation.shape[1] else np.asarray([], dtype=np.int64)
        correct = np.asarray(edge_correct, dtype=np.float64)
        support = np.asarray(edge_support, dtype=np.float64)
        support_weighted_correct = float(np.average(correct, weights=support)) if correct.size and support.sum() > 0 else 0.0
        pair_stats.append({
            "view_pair": [int(getattr(left_view, "source_view_index", -1)), int(getattr(right_view, "source_view_index", -1))],
            "source_ball_count": int(len(left)),
            "target_ball_count": int(len(right)),
            "jointly_observed_count": int(len(left_observed & right_observed)),
            "joint_observation_fraction_of_union": float(len(left_observed & right_observed) / len(left_observed | right_observed)) if left_observed or right_observed else 0.0,
            "edge_count": int(relation.sum()),
            "edge_density": float(relation.mean()) if relation.size else 0.0,
            "isolated_source_fraction": float((row_degree == 0).mean()) if row_degree.size else 0.0,
            "isolated_target_fraction": float((col_degree == 0).mean()) if col_degree.size else 0.0,
            "largest_component_fraction": _largest_bipartite_component_fraction(relation),
            "edge_support": _summary(edge_support),
            "edge_overlap_score": _summary(edge_overlap_score),
            "edge_min_endpoint_purity_diagnostic_only": _summary(edge_min_purity),
            "edge_max_endpoint_radius": _summary(edge_max_radius),
            "semantic_correct_edge_fraction_diagnostic_only": float(correct.mean()) if correct.size else 0.0,
            "support_weighted_semantic_correct_fraction_diagnostic_only": support_weighted_correct,
            "semantic_correct_vs_min_purity_spearman_diagnostic_only": _safe_spearman(edge_correct, edge_min_purity),
            "semantic_correct_vs_max_radius_spearman_diagnostic_only": _safe_spearman(edge_correct, edge_max_radius),
            "semantic_correct_vs_support_spearman_diagnostic_only": _safe_spearman(edge_correct, edge_support),
        })
    return {"threshold": float(threshold), "view_stats": view_stats, "view_pair_stats": pair_stats}


def summarize_observation_pattern_fusion(view_hs, view_masks, labels, n_clusters, seed, min_pattern_size=None):
    masks = view_masks.detach().cpu().numpy().astype(bool)
    labels = labels.detach().cpu().numpy().astype(np.int64).reshape(-1)
    views = [hidden.detach().cpu().numpy() for hidden in view_hs]
    stacked = np.stack(views, axis=1)
    denominator = masks.sum(axis=1, keepdims=True).clip(min=1)
    fused = (stacked * masks[:, :, None]).sum(axis=1) / denominator
    if min_pattern_size is None:
        min_pattern_size = max(50, 5 * int(n_clusters))

    patterns = []
    pattern_strings = np.asarray(["".join("1" if value else "0" for value in row) for row in masks])
    for pattern in sorted(np.unique(pattern_strings)):
        indices = np.flatnonzero(pattern_strings == pattern)
        available_views = [index for index, value in enumerate(pattern) if value == "1"]
        eligible = bool(len(indices) >= min_pattern_size and len(indices) >= n_clusters and available_views)
        record = {
            "pattern": pattern,
            "sample_count": int(len(indices)),
            "available_views": available_views,
            "is_multi_view_pattern": len(available_views) >= 2,
            "eligible": eligible,
            "min_pattern_size": int(min_pattern_size),
        }
        if eligible:
            subset_labels = labels[indices]
            record["class_count_diagnostic_only"] = int(np.unique(subset_labels).size)
            record["fused"] = _cluster_metrics(fused[indices], subset_labels, n_clusters, seed)
            record["single_views"] = {
                str(view): _cluster_metrics(views[view][indices], subset_labels, n_clusters, seed)
                for view in available_views
            }
            best_single_acc = max(value["ACC"] for value in record["single_views"].values())
            mean_single_acc = float(np.mean([value["ACC"] for value in record["single_views"].values()]))
            record["best_single_ACC_diagnostic_only"] = float(best_single_acc)
            record["mean_single_ACC"] = mean_single_acc
            record["fusion_minus_best_single_ACC_diagnostic_only"] = float(record["fused"]["ACC"] - best_single_acc)
            record["fusion_minus_mean_single_ACC"] = float(record["fused"]["ACC"] - mean_single_acc)
        patterns.append(record)

    eligible_multi = [item for item in patterns if item["eligible"] and item["is_multi_view_pattern"]]
    return {
        "min_pattern_size": int(min_pattern_size),
        "pattern_count": int(len(patterns)),
        "eligible_pattern_count": int(sum(item["eligible"] for item in patterns)),
        "eligible_multi_view_pattern_count": int(len(eligible_multi)),
        "fused_beats_best_single_count_diagnostic_only": int(sum(item["fusion_minus_best_single_ACC_diagnostic_only"] > 0 for item in eligible_multi)),
        "fused_beats_mean_single_count": int(sum(item["fusion_minus_mean_single_ACC"] > 0 for item in eligible_multi)),
        "patterns": patterns,
    }


def summarize_missing_mechanisms(mv_gblist, view_hs, view_masks, labels, n_clusters, seed, threshold=0.1, min_pattern_size=None):
    return {
        "labels_usage": "offline_diagnostic_only",
        "ball_relation_evidence": summarize_ball_and_relation_evidence(mv_gblist, threshold=threshold),
        "observation_pattern_fusion": summarize_observation_pattern_fusion(
            view_hs,
            view_masks,
            labels,
            n_clusters=n_clusters,
            seed=seed,
            min_pattern_size=min_pattern_size,
        ),
    }
