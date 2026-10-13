"""Construct one of the loss functions included in the public release."""

from granular.granular_loss import (
    MultiviewGCLoss,
    PermutedRelationMultiviewGCLoss,
)


SUPPORTED_MODES = {
    "standard",
    "permuted_relation_standard",
}


def build_granular_loss(ds_config):
    """Return the configured granular loss."""
    mode = ds_config.get("granular_loss_mode", "standard")
    if mode == "standard":
        return MultiviewGCLoss()
    if mode == "permuted_relation_standard":
        return PermutedRelationMultiviewGCLoss(
            relation_seed=ds_config.get("relation_control_seed", 0),
        )
    supported = ", ".join(sorted(SUPPORTED_MODES))
    raise ValueError(f"Unsupported granular_loss_mode {mode!r}; choose one of: {supported}")
