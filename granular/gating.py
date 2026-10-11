"""Mask path utilities used by the paper-path training loop.

Historical reliability / gating rule machinery lives in
``granular._exploration.gating_exploration`` and is intentionally not imported
here. Only the small, paper-path helpers remain.
"""


def parse_rho_from_mask_path(mask_path):
    if not mask_path:
        return None
    stem = str(mask_path).replace("\\", "/").split("/")[-1].rsplit(".", 1)[0]
    parts = stem.split("_")
    if "rho" not in parts:
        return None
    try:
        return float(parts[parts.index("rho") + 1])
    except (IndexError, ValueError):
        return None