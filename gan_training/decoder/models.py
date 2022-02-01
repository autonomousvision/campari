import torch
from torch import nn
import numpy as np

class ReluNetworkDisentangle(nn.Module):
    def __init__(self, in_features=3, hidden_features=128, hidden_layers=8, hidden_layers_view=1, c_dim=0,
                skips=[4], density_offset=0, **kwargs):
        super().__init__()

        self.c_dim = c_dim
        self.skips = skips
        self.actvn = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.density_offset = density_offset

        in_features = in_features * 2 * 10 + in_features
        in_features_view = 3 * 2 * 4 + 3

        # First block 
        self.fc_in = nn.Linear(in_features, hidden_features)
        self.blocks = nn.ModuleList([
            nn.Linear(hidden_features, hidden_features) for i in range(hidden_layers)
        ])
        if c_dim > 0:
            self.fc_c = nn.Linear(c_dim // 2, hidden_features)
        if len(skips) > 0:
            self.skips = nn.ModuleList([
                nn.Linear(in_features, hidden_features) for i in range(len(skips))
            ])
            if c_dim > 0:
                self.skips_c = nn.ModuleList([
                    nn.Linear(c_dim // 2, hidden_features) for i in range(len(skips))
                ])

        # Density head out
        self.fc_out = nn.Linear(hidden_features, 1)

        # View-dep block
        if c_dim > 0:
            self.fc_view_c = nn.Linear(c_dim // 2, hidden_features)
        self.net_view = []
        for i in range(hidden_layers_view):
            out_dim = hidden_features if i < hidden_layers_view - 1 else 3
            self.net_view.append(nn.ReLU(inplace=True))
            self.net_view.append(nn.Linear(hidden_features, out_dim))
        self.net_view = nn.Sequential(*self.net_view)
        # RGB out
        self.fc_view = nn.Linear(in_features_view, hidden_features)

    def encode_points(self, p, is_point=True):
        assert(
            (p.max() < np.pi) and
            (p.min() > -np.pi)
        )
        L = 10 if is_point else 4
        p_transformed = torch.cat([p] + [torch.cat(
            [torch.sin((2 ** i) * p),
                torch.cos((2 ** i) * p)],
            dim=-1) for i in range(L)], dim=-1)
        return p_transformed

    def forward(self, coords, condition=None, ray_dir=None, add_noise=True, activate_sigma=True, activate_rgb=True, raw_noise_std=1.):
        if condition is not None:
            c_dim = condition.shape[-1] // 2
            cond_shape, cond_app = condition[..., :c_dim], condition[..., c_dim:]

        # First block
        coords = self.encode_points(coords)
        net = self.fc_in(coords)
        if condition is not None and c_dim > 0:
            net = net + self.fc_c(cond_shape)
        n_skip = 0
        for idx, layer in enumerate(self.blocks):
            if idx in self.skips:
                net = net + self.skips[n_skip](coords)
                if condition is not None and c_dim > 0:
                    net = net + self.skips_c[n_skip](cond_shape) 
                n_skip += 1
            net = layer(self.actvn(net))

        # Density hehad out
        output = self.density_offset + self.fc_out(self.actvn(net))
        # if add_noise:
        output = output + torch.randn_like(output) * raw_noise_std
        if activate_sigma:
            output = self.actvn(output)

        # View-dep block
        if ray_dir is not None:
            ray_dir = self.encode_points(ray_dir / torch.norm(ray_dir, dim=-1, keepdim=True), False)
            net = net + self.fc_view(ray_dir)
        if condition is not None and c_dim > 0:
            net = net + self.fc_view_c(cond_app) 
        output2 = self.net_view(net)
        if activate_rgb:
            output2 = self.sigmoid(output2)
        output = torch.cat([output, output2], dim=-1)
        return output
