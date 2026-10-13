"""Granular-ball data structures used by the contrastive loss."""

import numpy as np
import torch
from sklearn.cluster import KMeans


class GranularBall:
    """A group of samples represented by its center and mean radius."""

    def __init__(self, data, labels=None, indices=None):
        self.data = data
        self.labels = labels
        self.indices = (
            np.arange(data.shape[0])
            if indices is None
            else np.asarray(indices).reshape(-1)
        )
        self.num_smp, self.dim = data.shape
        self.center = data.mean(dim=0)
        self.r = torch.norm(data - self.center, p=2, dim=1).mean()

    def split_balls(self, p):
        """Partition this ball into groups with approximately ``p`` samples."""
        data = self.data.detach().cpu().numpy()
        if p == 1:
            # Each observed sample must remain distinct, even for duplicate embeddings.
            assignments = np.arange(data.shape[0])
            num_balls = data.shape[0]
        else:
            num_balls = max(self.num_smp // p, 1)
            num_balls = min(num_balls, np.unique(data, axis=0).shape[0])
            assignments = KMeans(
                n_clusters=num_balls,
                n_init=10,
                random_state=42,
            ).fit_predict(data)

        assignments = torch.from_numpy(assignments).long()
        sub_balls = []
        for ball_index in range(num_balls):
            positions = torch.nonzero(assignments == ball_index, as_tuple=True)[0]
            tensor_positions = positions.to(self.data.device)
            sub_data = self.data[tensor_positions]
            sub_labels = (
                self.labels[tensor_positions] if self.labels is not None else None
            )
            sub_indices = self.indices[positions.numpy()]
            sub_balls.append(GranularBall(sub_data, sub_labels, sub_indices))
        return sub_balls, assignments


class GBList:
    """A collection of granular balls for one view."""

    def __init__(self, data, labels=None, p=8, indices=None):
        self.data = data
        self.labels = labels
        if indices is None:
            self.indices = np.arange(data.shape[0])
        else:
            source = indices.detach().cpu().numpy() if torch.is_tensor(indices) else indices
            self.indices = np.asarray(source).reshape(-1)
        initial_ball = GranularBall(data, labels, self.indices)
        self.granular_balls, self.y_parts = initial_ball.split_balls(p)

    def __len__(self):
        return len(self.granular_balls)

    def __getitem__(self, index):
        return self.granular_balls[index]

    def get_centers(self):
        """Return one center vector per granular ball."""
        return torch.vstack([ball.center for ball in self.granular_balls])


class MVGBList:
    """Build one granular-ball collection for each nonempty observed view."""

    def __init__(self, mv_data, labels=None, p=8, view_masks=None, sample_indices=None):
        self.gblists = []
        for view_index, view in enumerate(mv_data):
            if view_masks is None:
                view_data = view
                view_labels = labels
                view_indices = sample_indices
            else:
                observed = view_masks[:, view_index].bool()
                if not observed.any().item():
                    continue
                view_data = view[observed]
                view_labels = labels[observed] if labels is not None else None
                view_indices = (
                    sample_indices[observed] if sample_indices is not None else None
                )
            granular_balls = GBList(
                view_data,
                view_labels,
                p=p,
                indices=view_indices,
            )
            granular_balls.source_view_index = view_index
            self.gblists.append(granular_balls)

    def __len__(self):
        return len(self.gblists)

    def __getitem__(self, index):
        return self.gblists[index]
