"""V1 128-D multi-view autoencoder used by the paper path.

The exploration-only representations (consensus / residual split,
auxiliary residual, split-control) live in
:mod:`model._exploration.autoencoders_exploration` and are not imported by
this module.
"""
from typing import Sequence

import torch
import torch.nn as nn
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable


class Encoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 feature_dim: int,
                 middle_dims: Sequence[int] = (1024, 512, 256),
                 use_linear_projection: bool = False):
        super(Encoder, self).__init__()
        middle_dims = [input_dim] + list(middle_dims) + [feature_dim]
        middle_layers = nn.ModuleList()
        for i in range(len(middle_dims) - 2):
            layer = [nn.Linear(middle_dims[i], middle_dims[i + 1])]
            if not use_linear_projection:
                layer.extend([
                    nn.BatchNorm1d(middle_dims[i + 1]),
                    nn.ReLU(inplace=True)
                ])
            middle_layers.append(nn.Sequential(*layer))
        middle_layers.append(nn.Linear(middle_dims[-2], middle_dims[-1]))
        self.middle_layers = middle_layers
        self.middle_dims = middle_dims

    def forward(self, x):
        for layer in self.middle_layers:
            x = layer(x)
        return x


class Decoder(nn.Module):
    def __init__(self,
                 feature_dim: int,
                 output_dim: int,
                 middle_dims: Sequence[int] = (256, 512, 1024),
                 use_linear_projection: bool = False):
        super(Decoder, self).__init__()
        middle_dims = [feature_dim] + list(middle_dims) + [output_dim]
        middle_layers = nn.ModuleList()
        for i in range(len(middle_dims) - 2):
            layer = [nn.Linear(middle_dims[i], middle_dims[i + 1])]
            if not use_linear_projection:
                layer.extend([
                    nn.BatchNorm1d(middle_dims[i + 1]),
                    nn.ReLU(inplace=True)
                ])
            middle_layers.append(nn.Sequential(*layer))
        middle_layers.append(nn.Linear(middle_dims[-2], middle_dims[-1]))
        # 控制输出范围在0-1之间
        middle_layers.append(nn.Sigmoid())
        self.middle_layers = middle_layers
        self.middle_dims = middle_dims

    def forward(self, x):
        for layer in self.middle_layers:
            x = layer(x)
        return x


# 重构和Prediction都可以使用自编码器的结构
class AutoEncoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 feature_dim: int,
                 middle_dims: Sequence[int] = (1024, 512, 256),
                 use_linear_projection: bool = False):
        super(AutoEncoder, self).__init__()
        output_dim = input_dim
        self.encoder = Encoder(input_dim, feature_dim, middle_dims, use_linear_projection)
        self.decoder = Decoder(feature_dim, output_dim, middle_dims[::-1], use_linear_projection)

    def forward(self, x):
        hidden = self.encoder(x)
        x_rec = self.decoder(hidden)
        return hidden, x_rec


class MultiviewAutoEncoder(nn.Module):
    """Multi-view wrapper around per-view :class:`AutoEncoder`.

    The paper path always uses ``representation_mode='v1_128'`` (one
    128-D latent per view with a final L2 normalization layer appended
    by the caller).  Other representation modes used during the research
    phase have been removed from the public path.
    """

    def __init__(self, view_dims, latent_dim, middle_encoders, use_linear_projection=False,
                 representation_mode="v1_128"):
        super(MultiviewAutoEncoder, self).__init__()
        if representation_mode != "v1_128":
            raise ValueError(
                f"Public paper path supports only representation_mode='v1_128', got {representation_mode!r}."
            )
        num_view = len(view_dims)
        autoencoder_list = nn.ModuleList()
        for i in range(num_view):
            if middle_encoders is None:
                middle_dims = (1024, 512, 256)
            else:
                middle_dims = middle_encoders[i]
            autoencoder_list.append(
                AutoEncoder(
                    view_dims[i], latent_dim, middle_dims=middle_dims,
                    use_linear_projection=use_linear_projection,
                )
            )
        self.num_view = num_view
        self.view_dims = view_dims
        self.latent_dim = latent_dim
        self.representation_mode = representation_mode
        self.autoencoder_list = autoencoder_list

    def __getitem__(self, i):
        return self.autoencoder_list[i]

    def forward(self, x):
        """Multi-view latent features and reconstructions.

        :param x: list of per-view tensors, shape ``[num_view] * [batch, feature_dim_v]``
        :return: ``(hidden_list, x_rec_list)``
        """
        hidden_list, x_rs = [], []
        for i in range(self.num_view):
            autoencoder = self.autoencoder_list[i]
            hidden = autoencoder.encoder(x[i])
            x_r_view = autoencoder.decoder(hidden)
            hidden_list.append(hidden)
            x_rs.append(x_r_view)
        return hidden_list, x_rs

    def contrast_hidden(self, hidden_list):
        return hidden_list

    def cluster_hidden(self, hidden_list):
        return hidden_list


class Normalize(nn.Module):
    def __init__(self, p=2, dim=1):
        super(Normalize, self).__init__()
        self.normalize = nn.functional.normalize
        self.p = p
        self.dim = dim

    def forward(self, x):
        return self.normalize(x, p=self.p, dim=self.dim)