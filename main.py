"""Training entry point for Mask-Aware Granular-Ball Contrastive Clustering.

The public path of this module trains the V1 128-D multi-view autoencoder
with the standard cross-view granular-ball contrastive loss
(:class:`granular.granular_loss.MultiviewGCLoss`) or its permuted-relation
control, on either a full dataset, a zero-fill incomplete dataset, or a
masked incomplete dataset.  Reconstruction is masked per view; cross-view
ball construction uses only the observed samples per view; the relation is
constructed by co-observation identity.  Final clustering fuses each
sample's available views before KMeans.

"""
import argparse
import os.path as path

import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

try:
    from torchsummary import summary
except ImportError:
    def summary(*args, **kwargs):
        return None

from granular.base import MVGBList
from granular.loss_factory import build_granular_loss
from granular.missing_mechanism_diagnostics import summarize_missing_mechanisms
from model.autoencoder import MultiviewAutoEncoder, Normalize
from utils.common import init_torch, load_json
from utils.dataset import (
    MaskedMultiviewDataset,
    MultiviewDataset,
    ZeroFillMultiviewDataset,
)
from utils.score import cluster_metric


PROJECT_ROOT = path.dirname(path.abspath(__file__))


def _project_path(value):
    """Resolve a repository-relative path without depending on the current directory."""
    if value is None or path.isabs(value):
        return value
    return path.join(PROJECT_ROOT, value)


def run_on_setting(args, **kwargs):
    """Train and evaluate one (dataset, config) pair.

    Config precedence: explicit ``kwargs`` > dataset-specific config > global
    config.  Independent seed control is opt-in via ``data_seed`` /
    ``train_seed`` / ``eval_seed`` in ``kwargs``.
    """
    global_config_path = _project_path(args.global_config)
    global_params = load_json(global_config_path)
    if global_params["config"] is None:
        config_path = path.dirname(global_config_path)
        global_params["config"] = path.join(config_path, args.dataset + ".json")
    else:
        global_params["config"] = _project_path(global_params["config"])
    ds_config = load_json(global_params["config"])

    for key in global_params:
        if key not in ds_config:
            ds_config[key] = global_params[key]
    for key in kwargs:
        ds_config[key] = kwargs[key]

    independent_seed_control = any(
        key in kwargs for key in ("data_seed", "train_seed", "eval_seed")
    )
    data_seed = int(ds_config.get("data_seed", ds_config["seed"]))
    train_seed = int(ds_config.get("train_seed", ds_config["seed"]))
    eval_seed = int(ds_config.get("eval_seed", ds_config["seed"]))
    init_torch(seed=data_seed if independent_seed_control else ds_config["seed"])

    src = _project_path(ds_config["src"])
    ds_file = ds_config.get("file_name", args.dataset + ".mat")
    ds_path = path.join(src, ds_config["parent"], ds_file)
    missing_mode = ds_config.get("missing_mode")
    missing_mask_path = _project_path(ds_config.get("missing_mask_path"))
    missing_mechanism_diagnostics_enabled = bool(
        ds_config.get("missing_mechanism_diagnostics", False)
    )
    if missing_mechanism_diagnostics_enabled and missing_mode != "masked":
        raise ValueError("missing_mechanism_diagnostics requires missing_mode='masked'")
    if missing_mask_path is not None and missing_mode in ("zero_fill", "masked"):
        dataset_cls = MaskedMultiviewDataset if missing_mode == "masked" else ZeroFillMultiviewDataset
        mv_dataset = dataset_cls(
            ds_path,
            ds_config["device"],
            missing_mask_path,
            key_feature=ds_config["key_feature"],
            key_label=ds_config["key_label"],
            views=ds_config["select_views"],
            normalize=ds_config["normalize"],
        )
    else:
        mv_dataset = MultiviewDataset(
            ds_path,
            ds_config["device"],
            key_feature=ds_config["key_feature"],
            key_label=ds_config["key_label"],
            views=ds_config["select_views"],
            normalize=ds_config["normalize"],
        )

    if independent_seed_control:
        init_torch(seed=train_seed)
    batch_size = ds_config["batch_size"]
    if batch_size == -1:
        batch_size = len(mv_dataset)
    dataloader = DataLoader(mv_dataset, batch_size, shuffle=True, num_workers=0)

    representation_mode = ds_config.get("representation_mode", "v1_128")
    if representation_mode != "v1_128":
        raise ValueError(
            f"Public paper path supports only representation_mode='v1_128', got {representation_mode!r}."
        )
    singleton_ball_control = bool(ds_config.get("singleton_ball_control", False))
    if singleton_ball_control and int(ds_config["p"]) != 1:
        raise ValueError("singleton_ball_control requires p=1")
    mv_aes = MultiviewAutoEncoder(
        mv_dataset.view_dims,
        ds_config["latent_dim"],
        ds_config["autoencoder"]["mid_archs"],
        ds_config["use_linear_projection"],
        representation_mode="v1_128",
    )
    for v in range(mv_dataset.num_view):
        mv_aes[v].encoder.middle_layers.append(Normalize())
    mv_aes.to(ds_config["device"])
    summary(mv_aes[0], (mv_dataset.view_dims[0],))

    optimizer = torch.optim.Adam(
        mv_aes.parameters(),
        lr=ds_config["learning_rate"],
        weight_decay=ds_config["weight_decay"],
    )
    scheduler_t_max = ds_config.get("scheduler_t_max", ds_config["epochs"])
    if scheduler_t_max is None or scheduler_t_max < 1:
        scheduler_t_max = ds_config["epochs"]
    scheduler = CosineAnnealingLR(optimizer, T_max=scheduler_t_max, eta_min=0.0)

    weights = ds_config["loss_weights"]
    criterion_rec = nn.MSELoss()
    criterion_gra = build_granular_loss(ds_config)
    eval_interval = max(1, int(ds_config.get("eval_interval", 1)))

    kmeans = KMeans(
        n_clusters=mv_dataset.num_class,
        n_init=10,
        random_state=eval_seed if independent_seed_control else ds_config["seed"],
    )

    result = {
        "epoch": [],
        "loss_con": [],
        "loss_rec": [],
        "ACC": [],
        "NMI": [],
        "ARI": [],
        "PUR": [],
        "sh": [],
        "ch": [],
        "db": [],
        "missing_mechanism_diagnostics": [],
    }

    use_granular_training = ds_config["p"] > 1 or singleton_ball_control

    for epoch in range(ds_config["epochs"]):
        loss_con_avg = 0.0
        loss_rec_avg = 0.0
        mv_aes.train()
        for batch in dataloader:
            if missing_mode == "masked":
                x, y, obs_mask, sample_indices = batch
            else:
                x, y = batch
                obs_mask = None
                sample_indices = None

            reconstruction_hs, x_rs = mv_aes(x)
            hs = mv_aes.contrast_hidden(reconstruction_hs)

            loss_rec = torch.tensor(0.0, device=ds_config["device"])
            for v in range(mv_dataset.num_view):
                if obs_mask is None:
                    loss_rec = loss_rec + criterion_rec(x[v], x_rs[v])
                else:
                    observed = obs_mask[:, v].bool()
                    if observed.sum().item() > 0:
                        loss_rec = loss_rec + criterion_rec(x[v][observed], x_rs[v][observed])

            if use_granular_training and len(hs) >= 2:
                mv_gblist = MVGBList(
                    hs, labels=None, p=ds_config["p"],
                    view_masks=obs_mask, sample_indices=sample_indices,
                )
                if len(mv_gblist) >= 2:
                    loss_con = criterion_gra(mv_gblist)
                else:
                    loss_con = torch.tensor(0.0, device=ds_config["device"])
            else:
                loss_con = torch.tensor(0.0, device=ds_config["device"])

            loss = weights[0] * loss_con + weights[1] * loss_rec
            loss_con_avg += loss_con.item()
            loss_rec_avg += loss_rec.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_con_avg /= max(len(dataloader), 1)
        loss_rec_avg /= max(len(dataloader), 1)
        scheduler.step()

        if (epoch + 1) % eval_interval != 0 and epoch != ds_config["epochs"] - 1:
            print(f"epoch {epoch + 1}, loss_con {loss_con_avg:.4f}, loss_rec {loss_rec_avg:.4f}, eval skipped")
            continue

        mv_aes.eval()
        with torch.no_grad():
            reconstruction_view_hs, _ = mv_aes(mv_dataset.data)
            view_hs = mv_aes.cluster_hidden(reconstruction_view_hs)
            if missing_mode == "masked":
                stacked_hs = torch.stack(view_hs, dim=0)
                mask = mv_dataset.mask.T.unsqueeze(-1).to(stacked_hs.device)
                denom = mask.sum(dim=0).clamp_min(1.0)
                hs_eval = (stacked_hs * mask).sum(dim=0) / denom
                hs_eval = hs_eval.detach().cpu().numpy()
            else:
                hs_eval = torch.stack(view_hs, dim=0).mean(0).detach().cpu().numpy()

            y_pred = kmeans.fit_predict(hs_eval)
            y_true = mv_dataset.labels.cpu().numpy()
            acc, nmi, pur = cluster_metric(y_true, y_pred)
            acc, nmi, pur = round(acc, 4) * 100, round(nmi, 4) * 100, round(pur, 4) * 100
            ari = round(adjusted_rand_score(y_true, y_pred), 4) * 100
            sh = silhouette_score(hs_eval, y_pred, metric="euclidean")
            ch = calinski_harabasz_score(hs_eval, y_pred)
            db = davies_bouldin_score(hs_eval, y_pred)

            if missing_mechanism_diagnostics_enabled and epoch == ds_config["epochs"] - 1:
                sample_indices = torch.arange(len(mv_dataset), device=ds_config["device"])
                eval_gblist = MVGBList(
                    view_hs,
                    mv_dataset.labels,
                    ds_config["p"],
                    view_masks=mv_dataset.mask,
                    sample_indices=sample_indices,
                )
                mechanism_stats = summarize_missing_mechanisms(
                    eval_gblist,
                    view_hs,
                    mv_dataset.mask,
                    mv_dataset.labels,
                    n_clusters=mv_dataset.num_class,
                    seed=eval_seed if independent_seed_control else ds_config["seed"],
                    threshold=ds_config.get("granular_relation_threshold", 0.1),
                    min_pattern_size=ds_config.get("missing_mechanism_min_pattern_size"),
                )
                mechanism_stats["epoch"] = epoch
                mechanism_stats["dataset"] = args.dataset
                result["missing_mechanism_diagnostics"].append(mechanism_stats)
                fusion_stats = mechanism_stats["observation_pattern_fusion"]
                print(
                    f"missing_mechanism_diagnostics epoch {epoch + 1}, "
                    f"view_pairs {len(mechanism_stats['ball_relation_evidence']['view_pair_stats'])}, "
                    f"eligible_patterns {fusion_stats['eligible_pattern_count']}"
                )

        result["epoch"].append(epoch)
        result["loss_con"].append(loss_con_avg)
        result["loss_rec"].append(loss_rec_avg)
        result["ACC"].append(acc)
        result["NMI"].append(nmi)
        result["ARI"].append(ari)
        result["PUR"].append(pur)
        result["sh"].append(sh)
        result["ch"].append(ch)
        result["db"].append(db)
        print(
            f"epoch {epoch + 1}, loss_con {loss_con_avg:.4f}, loss_rec {loss_rec_avg:.4f}, "
            f"acc {acc:.2f}%, nmi {nmi:.2f}%, ari {ari:.2f}%, pur {pur:.2f}%, "
            f"sh {sh:.4f}, ch {ch:.2f}, db {db:.4f}"
        )

    return result


def run():
    """Run one dataset with its published configuration."""
    parser = argparse.ArgumentParser(description="Train MGBCC on one dataset.")
    parser.add_argument("--dataset", type=str, default="Caltech101-20")
    parser.add_argument(
        "--global_config",
        type=str,
        default=path.join(PROJECT_ROOT, "config", "granular_config", "global.json"),
        help="The path of the global config file.",
    )
    args = parser.parse_args()
    result = run_on_setting(args)
    final = {key: values[-1] for key, values in result.items() if values}
    print("final", final)


if __name__ == "__main__":
    run()
