import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.common import load_json, load_mat


def parse_csv_floats(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_ints(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def make_random_one_missing_mask(num_samples, num_views, incomplete_rate, seed):
    rng = np.random.default_rng(seed)
    mask = np.ones((num_samples, num_views), dtype=np.int8)
    num_incomplete = int(round(num_samples * incomplete_rate))
    selected = rng.choice(num_samples, size=num_incomplete, replace=False)
    missing_views = rng.integers(0, num_views, size=num_incomplete)
    mask[selected, missing_views] = 0
    return mask


def summarize_mask(mask, incomplete_rate, seed, dataset, protocol):
    num_samples, num_views = mask.shape
    incomplete_sample_count = int(np.sum(np.any(mask == 0, axis=1)))
    missing_entry_count = int(np.sum(mask == 0))
    return {
        "dataset": dataset,
        "protocol": protocol,
        "seed": seed,
        "num_samples": num_samples,
        "num_views": num_views,
        "target_incomplete_sample_rate": incomplete_rate,
        "actual_incomplete_sample_rate": incomplete_sample_count / num_samples,
        "missing_entry_rate": missing_entry_count / (num_samples * num_views),
        "incomplete_sample_count": incomplete_sample_count,
        "missing_entry_count": missing_entry_count,
        "min_observed_views_per_sample": int(mask.sum(axis=1).min()),
        "max_observed_views_per_sample": int(mask.sum(axis=1).max()),
        "missing_rate_per_view": np.mean(mask == 0, axis=0).tolist(),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate reproducible incomplete-view masks.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--global_config", default="./config/granular_config/global.json")
    parser.add_argument("--rates", default="0.1,0.3,0.5,0.7")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--out_root", default="./experiments/masks")
    args = parser.parse_args()

    global_params = load_json(args.global_config)
    config_path = global_params["config"]
    if config_path is None:
        config_path = os.path.join(os.path.dirname(args.global_config), args.dataset + ".json")
    ds_config = load_json(config_path)
    for key, value in global_params.items():
        ds_config.setdefault(key, value)

    ds_file = ds_config.get("file_name", args.dataset + ".mat")
    ds_path = os.path.join(ds_config["src"], ds_config["parent"], ds_file)
    data, labels = load_mat(
        ds_path,
        views=ds_config["select_views"],
        key_feature=ds_config["key_feature"],
        key_label=ds_config["key_label"],
    )
    num_samples = len(labels)
    num_views = len(data)

    rates = parse_csv_floats(args.rates)
    seeds = parse_csv_ints(args.seeds)
    protocol = "random_one_missing_view"
    out_dir = os.path.join(args.out_root, args.dataset, protocol)
    os.makedirs(out_dir, exist_ok=True)

    manifest = []
    for rate in rates:
        for seed in seeds:
            mask = make_random_one_missing_mask(num_samples, num_views, rate, seed)
            metadata = summarize_mask(mask, rate, seed, args.dataset, protocol)
            stem = f"rho_{rate:.1f}_seed_{seed}"
            npz_path = os.path.join(out_dir, stem + ".npz")
            json_path = os.path.join(out_dir, stem + ".json")
            np.savez_compressed(npz_path, mask=mask)
            with open(json_path, "w", encoding="utf-8") as fp:
                json.dump(metadata, fp, indent=2, ensure_ascii=False)
            manifest.append({"mask": npz_path, "metadata": json_path, **metadata})

    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2, ensure_ascii=False)

    print("dataset", args.dataset)
    print("samples", num_samples)
    print("views", num_views)
    print("generated", len(manifest))
    print("manifest", manifest_path)


if __name__ == "__main__":
    main()
