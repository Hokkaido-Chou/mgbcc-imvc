"""PyTorch datasets for complete and incomplete multi-view data."""

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.common import load_mat


class MultiviewDataset(Dataset):
    """Load a complete multi-view dataset into device tensors."""

    def __init__(
        self,
        root: str,
        device,
        key_feature="data",
        key_label="labels",
        normalize=True,
        views=None,
    ):
        data, labels = load_mat(
            root,
            views,
            key_feature=key_feature,
            key_label=key_label,
        )
        for index, view in enumerate(data):
            tensor = torch.as_tensor(view, dtype=torch.float32, device=device)
            if normalize:
                maximum = tensor.max(dim=0, keepdim=True).values
                minimum = tensor.min(dim=0, keepdim=True).values
                tensor = (tensor - minimum) / (maximum - minimum + 1e-12)
            data[index] = tensor

        self.data = data
        self.num_view = len(data)
        self.view_dims = [view.shape[1] for view in data]
        self.labels = torch.as_tensor(labels, dtype=torch.int32, device=device).reshape(-1)
        self.num_class = len(torch.unique(self.labels))
        self.device = device

    def __getitem__(self, index):
        return [view[index] for view in self.data], self.labels[index]

    def __len__(self):
        return len(self.labels)


class ZeroFillMultiviewDataset(MultiviewDataset):
    """Apply a missing-view mask by replacing unavailable views with zeros."""

    def __init__(
        self,
        root: str,
        device,
        mask_path: str,
        key_feature="data",
        key_label="labels",
        normalize=True,
        views=None,
    ):
        super().__init__(
            root,
            device,
            key_feature=key_feature,
            key_label=key_label,
            normalize=normalize,
            views=views,
        )
        with np.load(mask_path) as archive:
            mask = archive["mask"].astype(np.float32)
        expected_shape = (len(self.labels), self.num_view)
        if mask.shape != expected_shape:
            raise ValueError(
                f"Mask shape {mask.shape} does not match data shape {expected_shape}"
            )

        self.mask = torch.as_tensor(mask, dtype=torch.float32, device=device)
        for index in range(self.num_view):
            self.data[index] = self.data[index] * self.mask[:, index : index + 1]


class MaskedMultiviewDataset(ZeroFillMultiviewDataset):
    """Return the observation mask and original sample index with each item."""

    def __getitem__(self, index):
        views, label = super().__getitem__(index)
        sample_index = torch.as_tensor(index, dtype=torch.long, device=self.device)
        return views, label, self.mask[index], sample_index
