import torch
import numpy as np
from torch.utils.data import Dataset
from utils.common import load_mat


class MultiviewDataset(Dataset):
    def __init__(self, root: str, device,
                 key_feature="data", key_label="labels",
                 normalize=True, views=None):
        """
        :param root: 数据集路径
        :param device: cpu or cuda
        :param key_feature: 存储特征的关键字
        :param key_label: 存储标签的关键字
        :param views: 指定读取哪些视图
        """
        # load_scene --- customize load_<dataset> function if not consistent
        data, labels = load_mat(root, views, key_feature=key_feature, key_label=key_label)
        num_view = len(data)
        view_dims = [0] * num_view
        for i in range(num_view):
            # .to(device)
            data[i] = torch.as_tensor(data[i], dtype=torch.float32).to(device)
            if normalize:
                max_value, _ = torch.max(data[i], dim=0, keepdim=True)
                min_value, _ = torch.min(data[i], dim=0, keepdim=True)
                data[i] = (data[i] - min_value) / (max_value - min_value + 1e-12)
                # data[i] = (data[i] - torch.min(data[i])) / (torch.max(data[i]) - torch.min(data[i]) + 1e-12)
            view_dims[i] = data[i].shape[1]
        self.data = data
        self.num_view = num_view
        self.view_dims = view_dims
        # .to(device)
        self.labels = torch.as_tensor(labels, dtype=torch.int32).view((-1,)).to(device)
        self.num_class = len(torch.unique(self.labels))
        self.device = device

    def __getitem__(self, index):
        item = []
        for i in range(self.num_view):
            # .to(self.device)
            item.append(self.data[i][index])
        # .to(self.device)
        return item, self.labels[index]

    def __len__(self):
        return len(self.labels)


class ZeroFillMultiviewDataset(Dataset):
    def __init__(self, root: str, device, mask_path: str,
                 key_feature="data", key_label="labels",
                 normalize=True, views=None):
        data, labels = load_mat(root, views, key_feature=key_feature, key_label=key_label)
        mask = np.load(mask_path)["mask"].astype(np.float32)
        num_view = len(data)
        if mask.shape != (len(labels), num_view):
            raise ValueError(f"Mask shape {mask.shape} does not match data shape {(len(labels), num_view)}")

        view_dims = [0] * num_view
        for i in range(num_view):
            data[i] = torch.as_tensor(data[i], dtype=torch.float32).to(device)
            if normalize:
                max_value, _ = torch.max(data[i], dim=0, keepdim=True)
                min_value, _ = torch.min(data[i], dim=0, keepdim=True)
                data[i] = (data[i] - min_value) / (max_value - min_value + 1e-12)
            data[i] = data[i] * torch.as_tensor(mask[:, i:i + 1], dtype=torch.float32).to(device)
            view_dims[i] = data[i].shape[1]

        self.data = data
        self.mask = torch.as_tensor(mask, dtype=torch.float32).to(device)
        self.num_view = num_view
        self.view_dims = view_dims
        self.labels = torch.as_tensor(labels, dtype=torch.int32).view((-1,)).to(device)
        self.num_class = len(torch.unique(self.labels))
        self.device = device

    def __getitem__(self, index):
        item = []
        for i in range(self.num_view):
            item.append(self.data[i][index])
        return item, self.labels[index]

    def __len__(self):
        return len(self.labels)


class MaskedMultiviewDataset(ZeroFillMultiviewDataset):
    def __getitem__(self, index):
        item = []
        for i in range(self.num_view):
            item.append(self.data[i][index])
        return item, self.labels[index], self.mask[index], torch.as_tensor(index, dtype=torch.long, device=self.device)


