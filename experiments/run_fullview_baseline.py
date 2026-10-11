"""Run one fixed paper-path MGBCC configuration and save a JSON record."""

import argparse
import json
import os
import sys
import time
from argparse import Namespace
from datetime import datetime


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from main import run_on_setting


DIAGNOSTIC_HISTORY_KEYS = {"missing_mechanism_diagnostics"}


def to_jsonable(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    return value


def parse_args():
    parser = argparse.ArgumentParser(description="Run one fixed paper-path MGBCC configuration.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--global_config", default="./config/granular_config/global.json")
    parser.add_argument("--out_dir", default="./experiments/fullview_baseline")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--data_seed", type=int, default=None)
    parser.add_argument("--train_seed", type=int, default=None)
    parser.add_argument("--eval_seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--latent_dim", type=int, default=None)
    parser.add_argument("--p", type=int, default=None)
    parser.add_argument("--singleton_ball_control", choices=["true", "false"], default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--eval_interval", type=int, default=None)
    parser.add_argument("--scheduler_t_max", type=int, default=None)
    parser.add_argument("--normalize", choices=["true", "false"], default=None)
    parser.add_argument("--missing_mode", choices=["masked", "zero_fill"], default=None)
    parser.add_argument("--missing_mask_path", default=None)
    parser.add_argument(
        "--granular_loss_mode",
        choices=["standard", "permuted_relation_standard"],
        default=None,
    )
    parser.add_argument("--relation_control_seed", type=int, default=None)
    parser.add_argument("--missing_mechanism_diagnostics", choices=["true", "false"], default=None)
    parser.add_argument("--missing_mechanism_min_pattern_size", type=int, default=None)
    parser.add_argument("--granular_relation_threshold", type=float, default=None)
    parser.add_argument("--representation_mode", choices=["v1_128"], default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    overrides = {"device": args.device}
    for key in [
        "seed",
        "data_seed",
        "train_seed",
        "eval_seed",
        "epochs",
        "batch_size",
        "latent_dim",
        "p",
        "learning_rate",
        "eval_interval",
        "scheduler_t_max",
        "missing_mode",
        "missing_mask_path",
        "granular_loss_mode",
        "relation_control_seed",
        "missing_mechanism_min_pattern_size",
        "granular_relation_threshold",
        "representation_mode",
    ]:
        value = getattr(args, key)
        if value is not None:
            overrides[key] = value
    if args.singleton_ball_control is not None:
        overrides["singleton_ball_control"] = args.singleton_ball_control == "true"
    if args.normalize is not None:
        overrides["normalize"] = args.normalize == "true"
    if args.missing_mechanism_diagnostics is not None:
        overrides["missing_mechanism_diagnostics"] = args.missing_mechanism_diagnostics == "true"

    run_args = Namespace(dataset=args.dataset, global_config=args.global_config)
    started_at = datetime.now().isoformat(timespec="seconds")
    wall_started = time.perf_counter()
    result = run_on_setting(run_args, **overrides)
    wall_time_seconds = time.perf_counter() - wall_started
    finished_at = datetime.now().isoformat(timespec="seconds")

    final = {
        key: values[-1]
        for key, values in result.items()
        if values and key not in DIAGNOSTIC_HISTORY_KEYS
    }
    best_idx = max(range(len(result["ACC"])), key=lambda index: result["ACC"][index])
    best_debug = {
        key: values[best_idx] if len(values) == len(result["ACC"]) else values[-1]
        for key, values in result.items()
        if values and key not in DIAGNOSTIC_HISTORY_KEYS
    }

    payload = to_jsonable({
        "dataset": args.dataset,
        "started_at": started_at,
        "finished_at": finished_at,
        "wall_time_seconds": wall_time_seconds,
        "overrides": overrides,
        "reporting_rule": "Use final only for formal unsupervised comparison; best_debug is diagnostic.",
        "final": final,
        "best_debug": best_debug,
        "history": result,
    })

    os.makedirs(args.out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.out_dir, f"{args.dataset}_{stamp}.json")
    with open(out_path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
    print("saved_result", out_path)
    print("final", final)
    print("best_debug", best_debug)


if __name__ == "__main__":
    main()
