"""Shared configuration, random-seed, and MATLAB data-loading helpers."""

import json
import random
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy import io as scipy_io


def init_torch(seed):
    """Seed Python, NumPy, and PyTorch random-number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def load_json(file_path, encoding="utf-8"):
    with open(file_path, "r", encoding=encoding) as stream:
        return json.load(stream)


def _load_hdf5_mat(file_path, key_feature="data", key_label="labels"):
    with h5py.File(file_path, "r") as stream:
        labels = np.asarray(stream[key_label]).reshape(-1)
        num_samples = labels.size
        features = []
        for reference in np.asarray(stream[key_feature]).reshape(-1):
            view = np.asarray(stream[reference]).squeeze()
            if view.ndim == 1:
                view = view.reshape(num_samples, -1)
            elif view.shape[0] != num_samples and view.shape[-1] == num_samples:
                view = view.T
            features.append(view)
    return features, labels


def _load_numbered_view_mat(data, key_label="Y"):
    features = []
    view_index = 1
    while f"X{view_index}" in data:
        view = data[f"X{view_index}"].squeeze()
        if view.ndim > 2:
            view = view.reshape(view.shape[0], -1)
        features.append(view)
        view_index += 1
    if not features:
        raise KeyError("No numbered view keys such as X1 and X2 were found.")
    return features, data[key_label].reshape(-1)


def load_mat(file_path, views=None, key_feature="data", key_label="labels"):
    """Load, shuffle, and select views from a MATLAB multi-view dataset."""
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(
            f"Dataset file not found: {file_path}. "
            "Place the prepared .mat file under data/smalldata/."
        )

    try:
        data = scipy_io.loadmat(file_path)
    except (NotImplementedError, ValueError):
        features, labels = _load_hdf5_mat(
            file_path,
            key_feature=key_feature,
            key_label=key_label,
        )
    else:
        if key_feature in data and key_label in data:
            labels = data[key_label].reshape(-1)
            features = [
                data[key_feature][index][0].squeeze()
                for index in range(len(data[key_feature]))
            ]
        elif key_label in data:
            features, labels = _load_numbered_view_mat(data, key_label=key_label)
        elif "Y" in data:
            features, labels = _load_numbered_view_mat(data, key_label="Y")
        else:
            raise KeyError(f"Could not find feature and label keys in {file_path}")

    permutation = np.random.permutation(labels.size)
    features = [view[permutation] for view in features]
    labels = labels[permutation]
    selected_views = list(range(len(features))) if not views else views
    return [features[index] for index in selected_views], labels
