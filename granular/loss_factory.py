"""Build the cross-view granular-ball contrastive loss from a config.

This module is the single point that maps a ``granular_loss_mode`` config value
to a concrete ``torch.nn.Module`` instance.  Paper-path modes
(``"standard"``, ``"permuted_relation_standard"``) are wired directly to the
two losses in :mod:`granular.granular_loss`.  All exploration modes are
loaded **lazily** from :mod:`granular._exploration.losses_exploration` (and
sibling exploration modules) so the paper-path import graph stays small.
"""
from granular.granular_loss import (
    MultiviewGCLoss,
    PermutedRelationMultiviewGCLoss,
)


# Modes supported by the public paper path.  Anything else is an exploration
# mode and is looked up dynamically in the exploration package.
PAPER_MODES = {
    "standard",
    "permuted_relation_standard",
}


def _exploration_loss(mode, ds_config):
    """Lazy-import and instantiate an exploration loss.

    Raises a clear error if the exploration package cannot satisfy ``mode``.
    """
    try:
        from granular._exploration import losses_exploration
    except ImportError as exc:  # pragma: no cover - defensive
        raise RuntimeError(
            f"granular_loss_mode {mode!r} requires the exploration package"
        ) from exc

    if mode == "mutual_topology_standard":
        from granular._exploration.topology_loss_exploration import MutualTopologyMultiviewGCLoss
        return MutualTopologyMultiviewGCLoss(
            k_neighbors=ds_config.get("topology_k_neighbors", 10),
            min_shared_neighbors=ds_config.get("topology_min_shared_neighbors", 3),
        )

    if mode == "reliability":
        return losses_exploration.ReliabilityMultiviewGCLoss(
            support_ref=ds_config.get("reliability_support_ref", 3.0),
            use_support=ds_config.get("reliability_use_support", True),
            use_compactness=ds_config.get("reliability_use_compactness", True),
            use_degree=ds_config.get("reliability_use_degree", True),
            weight_floor=ds_config.get("reliability_weight_floor", 0.0),
        )

    if mode == "conservative_local":
        return losses_exploration.ConservativeLocalReliabilityMultiviewGCLoss(
            support_ref=ds_config.get("reliability_support_ref", 3.0),
            use_support=ds_config.get("reliability_use_support", True),
            use_compactness=ds_config.get("reliability_use_compactness", True),
            use_overlap=ds_config.get("local_reliability_use_overlap", True),
            modulation=ds_config.get("local_reliability_modulation", 0.25),
            relation_threshold=ds_config.get("granular_relation_threshold", 0.1),
            center_mode=ds_config.get("local_reliability_center_mode", "absolute"),
        )

    if mode == "anchor_guided_standard":
        return losses_exploration.AnchorGuidedMultiviewGCLoss(
            anchor_view_index=ds_config.get("anchor_view_index", 0),
            non_anchor_pair_weight=ds_config.get("anchor_non_anchor_pair_weight", 0.25),
        )

    if mode in {
        "anchor_consistent_standard",
        "anchor_adaptive_consistent_standard",
        "anchor_soft_bridge_standard",
        "anchor_soft_bridge_graph_standard",
        "anchor_soft_bridge_graph_budget_standard",
        "anchor_ratio_policy_consistent_standard",
        "anchor_filtered_consistent_standard",
        "anchor_conf_fixed_consistent_standard",
        "anchor_proto_consistent_standard",
        "anchor_false_negative_consistent_standard",
        "anchor_bridge_fn_consistent_standard",
        "anchor_graph_consistent_standard",
        "anchor_graph_scheduled_standard",
        "anchor_reconstruct_filter_standard",
        "anchor_reconstruct_split_standard",
        "anchor_reconstruct_aux_standard",
    }:
        anchor_kwargs = {
            "anchor_view_index": ds_config.get("anchor_view_index", 0),
            "anchor_bridge_weight": ds_config.get("anchor_bridge_weight", 0.5),
        }
        cls_map = {
            "anchor_consistent_standard": losses_exploration.AnchorConsistentMultiviewGCLoss,
            "anchor_adaptive_consistent_standard": losses_exploration.AnchorAdaptiveConsistentMultiviewGCLoss,
            "anchor_soft_bridge_standard": losses_exploration.AnchorSoftBridgeMultiviewGCLoss,
            "anchor_soft_bridge_graph_standard": losses_exploration.AnchorSoftBridgeGraphMultiviewGCLoss,
            "anchor_soft_bridge_graph_budget_standard": losses_exploration.AnchorSoftBridgeGraphBudgetMultiviewGCLoss,
            "anchor_ratio_policy_consistent_standard": losses_exploration.AnchorRatioPolicyConsistentMultiviewGCLoss,
            "anchor_filtered_consistent_standard": losses_exploration.AnchorFilteredConsistentMultiviewGCLoss,
            "anchor_conf_fixed_consistent_standard": losses_exploration.AnchorConfidenceFixedConsistentMultiviewGCLoss,
            "anchor_proto_consistent_standard": losses_exploration.AnchorPrototypeConsistentMultiviewGCLoss,
            "anchor_false_negative_consistent_standard": losses_exploration.AnchorFalseNegativeConsistentMultiviewGCLoss,
            "anchor_bridge_fn_consistent_standard": losses_exploration.AnchorBridgeFalseNegativeMultiviewGCLoss,
            "anchor_graph_consistent_standard": losses_exploration.AnchorGraphConsistentMultiviewGCLoss,
            "anchor_graph_scheduled_standard": losses_exploration.AnchorGraphScheduledConsistentMultiviewGCLoss,
            "anchor_reconstruct_filter_standard": losses_exploration.AnchorReconstructFilterMultiviewGCLoss,
            "anchor_reconstruct_split_standard": losses_exploration.AnchorReconstructSplitMultiviewGCLoss,
            "anchor_reconstruct_aux_standard": losses_exploration.AnchorReconstructAuxMultiviewGCLoss,
        }
        return cls_map[mode](**anchor_kwargs)

    raise ValueError(f"Unknown granular_loss_mode {mode!r}")


def build_granular_loss(ds_config):
    """Return ``(criterion_granular, criterion_standard_or_none)``.

    The paper path uses ``MultiviewGCLoss`` directly and ignores the standard
    companion.  Exploration modes that compare against a standard baseline
    return a tuple of (main, standard).
    """
    mode = ds_config.get("granular_loss_mode", "standard")

    if mode == "standard":
        return MultiviewGCLoss(), None

    if mode == "permuted_relation_standard":
        return PermutedRelationMultiviewGCLoss(
            relation_seed=ds_config.get("relation_control_seed", 0),
        ), None

    criterion = _exploration_loss(mode, ds_config)
    # Exploration modes that have a "standard" companion (used as a fallback
    # during gating / selector phases) are returned alongside MultiviewGCLoss.
    if mode in {
        "reliability",
        "conservative_local",
        "anchor_adaptive_consistent_standard",
        "anchor_filtered_consistent_standard",
        "anchor_conf_fixed_consistent_standard",
        "anchor_proto_consistent_standard",
        "anchor_false_negative_consistent_standard",
        "anchor_bridge_fn_consistent_standard",
        "anchor_graph_consistent_standard",
        "anchor_graph_scheduled_standard",
        "anchor_reconstruct_filter_standard",
        "anchor_reconstruct_split_standard",
        "anchor_reconstruct_aux_standard",
    }:
        return criterion, MultiviewGCLoss()

    return criterion, None