"""Self-contained one-epoch incomplete-view smoke test."""

from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.io import savemat

from main import PROJECT_ROOT, run_on_setting


def make_tiny_dataset(file_path):
    """Write two deterministic views of three compact synthetic clusters."""
    rng = np.random.default_rng(42)
    labels = np.repeat(np.arange(3), 10)
    centers_view1 = rng.normal(size=(3, 8)) * 3.0
    centers_view2 = rng.normal(size=(3, 6)) * 3.0
    view1 = centers_view1[labels] + rng.normal(scale=0.2, size=(30, 8))
    view2 = centers_view2[labels] + rng.normal(scale=0.2, size=(30, 6))
    savemat(file_path, {"X1": view1, "X2": view2, "Y": labels})


if __name__ == "__main__":
    config_path = Path(PROJECT_ROOT) / "config" / "granular_config" / "global.json"
    args = Namespace(dataset="Tiny", global_config=str(config_path))
    with TemporaryDirectory() as temporary_directory:
        dataset_path = Path(temporary_directory) / "Tiny.mat"
        mask_path = Path(temporary_directory) / "mask.npz"
        make_tiny_dataset(dataset_path)
        mask = np.ones((30, 2), dtype=np.int8)
        mask[::3, 0] = 0
        mask[1::3, 1] = 0
        np.savez_compressed(mask_path, mask=mask)
        result = run_on_setting(
            args,
            src=temporary_directory,
            parent="",
            file_name="Tiny.mat",
            device="cpu",
            epochs=1,
            batch_size=-1,
            latent_dim=4,
            p=2,
            learning_rate=1e-3,
            missing_mode="masked",
            missing_mask_path=str(mask_path),
        )
    print("Tiny test ACC:", result["ACC"])
