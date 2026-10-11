import random
import numpy as np
import torch
import json
import itertools

try:
    import hdf5storage as hdf
except ImportError:
    from scipy import io as hdf


def init_torch(seed):
    # 随机数种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def load_json(path,  encoding='utf-8'):
    with open(path, "r", encoding=encoding) as fp:
        params = json.load(fp)
    return params


def _load_hdf5_mat(path, key_feature="data", key_label="labels"):
    import h5py

    with h5py.File(path, "r") as fp:
        label = np.asarray(fp[key_label]).reshape((-1,))
        num_smp = label.size
        feature = []
        data_ref = np.asarray(fp[key_feature]).reshape((-1,))
        for ref in data_ref:
            tmp = np.asarray(fp[ref]).squeeze()
            if tmp.ndim == 1:
                tmp = tmp.reshape((num_smp, -1))
            elif tmp.shape[0] != num_smp and tmp.shape[-1] == num_smp:
                tmp = tmp.T
            feature.append(tmp)
    return feature, label


def _load_numbered_view_mat(data, key_label="Y"):
    feature = []
    view_idx = 1
    while f"X{view_idx}" in data:
        tmp = data[f"X{view_idx}"].squeeze()
        if tmp.ndim > 2:
            tmp = tmp.reshape((tmp.shape[0], -1))
        feature.append(tmp)
        view_idx += 1
    if not feature:
        raise KeyError("No numbered view keys like X1, X2 were found.")
    label = data[key_label].reshape((-1,))
    return feature, label


def load_mat(path, views=None, key_feature="data", key_label="labels"):
    try:
        data = hdf.loadmat(path)
        if key_feature in data and key_label in data:
            feature = []
            num_view = len(data[key_feature])
            label = data[key_label].reshape((-1,))
            for v in range(num_view):
                tmp = data[key_feature][v][0].squeeze()
                feature.append(tmp)
        elif key_label in data:
            feature, label = _load_numbered_view_mat(data, key_label=key_label)
            num_view = len(feature)
        elif "Y" in data:
            feature, label = _load_numbered_view_mat(data, key_label="Y")
            num_view = len(feature)
        else:
            raise KeyError(f"Could not find feature/label keys in {path}")
    except NotImplementedError:
        feature, label = _load_hdf5_mat(path, key_feature=key_feature, key_label=key_label)
        num_view = len(feature)
    num_smp = label.size
    # 打乱样本
    rand_permute = np.random.permutation(num_smp)
    for v in range(num_view):
        feature[v] = feature[v][rand_permute]
    label = label[rand_permute]
    if views is None or len(views) == 0:
        views = list(range(num_view))
    views_feature = [feature[v] for v in views]
    return views_feature, label


# 实现功能：返回参数组合的笛卡尔集
def get_all_parameters(*parameters_range):
    return itertools.product(*parameters_range)

