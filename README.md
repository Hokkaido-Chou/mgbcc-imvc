# Mask-Aware Granular-Ball Contrastive Clustering

Minimal source release for incomplete multi-view clustering. This branch contains the training code, dataset configurations, a fixed-run entry point, a missing-view mask generator, and a smoke test. It contains no datasets, masks, experimental logs, or result records.

## Install

Tested locally with Python 3.10. Install the paper-path dependencies:

```bash
pip install -r requirements.txt
```

## Datasets

Place prepared `.mat` files in `data/smalldata/` under the repository root. The raw source folders used locally for the four derived subsets are in the sibling `../data/` directory; they are not part of this release. `DATA_MANIFEST.csv` records the expected filenames, sizes, sample/view/class counts, and SHA-256 hashes.

| Config | File | Raw source in the checked local workspace |
|---|---|---|
| BDGP | `BDGP.mat` | Prepared benchmark file only |
| Caltech101-20 | `Caltech101-20.mat` | Prepared benchmark file only |
| MNIST-USPS | `MNIST_USPS.mat` | Prepared benchmark file only |
| GLC24-PA-Subset | `GLC24-PA-Subset.mat` | `../data/archive/` |
| MIRFlickr-25K-Subset | `MIRFlickr-25K-Subset.mat` | `../data/mirflickr25k/` and `../data/mirflickr25k_annotations_v080/` |
| N24News-Subset | `N24News-Subset.mat` | `../data/N24News/` |
| Semi-iNat-2021-Subset | `Semi-iNat-2021-Subset.mat` | `../data/semi-inat-2021/` |
| Tiny smoke test | `Tiny.mat` | Local test fixture |

The benchmark files and Tiny fixture must be obtained separately. Verify copied files with `sha256sum data/smalldata/<file>.mat` or PowerShell `Get-FileHash data/smalldata/<file>.mat -Algorithm SHA256`.

## Run

Run from the repository root. The smoke test needs `data/smalldata/Tiny.mat`:

```bash
python test.py
```

For a dataset already placed under `data/smalldata/`, generate deterministic missing-view masks, then run one configuration:

```bash
python experiments/generate_missing_masks.py --dataset BDGP --rates 0.5 --seeds 0
python experiments/run_fullview_baseline.py --dataset BDGP --device cpu --seed 0 --epochs 50 --eval_interval 5 --scheduler_t_max 50 --missing_mode masked --missing_mask_path experiments/masks/BDGP/random_one_missing_view/rho_0.5_seed_0.npz --granular_loss_mode standard --p 2 --latent_dim 128 --representation_mode v1_128
```

The mask generator writes to `experiments/masks/`, which Git ignores. Other supported dataset names are the configuration filenames under `config/granular_config/` without `.json`. The default `main.py` entry point performs a large parameter sweep; use `experiments/run_fullview_baseline.py` for a fixed run.

## Included files

- `main.py`, `granular/`, `model/`, `utils/`: paper-path training and evaluation code.
- `config/granular_config/`: global, seven dataset, and Tiny configurations.
- `experiments/run_fullview_baseline.py`: fixed-configuration runner.
- `experiments/generate_missing_masks.py`: deterministic missing-view mask generator.
- `test.py`: one-epoch Tiny smoke test.
- `DATA_MANIFEST.csv`: local dataset verification metadata; no dataset content.

The prepared datasets, raw data, generated masks, third-party official baselines, exploration code, experiment histories, reports, figures, and manuscript files are intentionally outside this minimal branch.

## Citation and license

See `CITATION.cff` and `LICENSE`.
