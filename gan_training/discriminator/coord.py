from torch import nn
import numpy as np
import torch
import kornia
import torch.nn.functional as F


def get_grid(x_dim=32, y_dim=32, to_cuda=True):
    xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
    yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

    xx_channel = xx_channel.float() / (x_dim - 1)
    yy_channel = yy_channel.float() / (y_dim - 1)

    xx_channel = xx_channel * 2 - 1
    yy_channel = yy_channel * 2 - 1

    xx_channel = xx_channel.repeat(1, 1, 1, 1).transpose(2, 3)
    yy_channel = yy_channel.repeat(1, 1, 1, 1).transpose(2, 3)
    if to_cuda:
        xx_channel = xx_channel.cuda()
        yy_channel = yy_channel.cuda()
    return xx_channel, yy_channel


class AddCoords(nn.Module):

    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

        self.grid4 = get_grid(4, 4)
        self.grid8 = get_grid(8, 8)
        self.grid16 = get_grid(16, 16)
        self.grid_32 = get_grid()
        self.grid_64 = get_grid(64, 64)
        self.grid_128 = get_grid(128, 128)

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()
        if x_dim == 4:
            xx_channel, yy_channel = self.grid4
        elif x_dim == 8:
            xx_channel, yy_channel = self.grid8
        elif x_dim == 16:
            xx_channel, yy_channel = self.grid16
        elif x_dim == 32:
            xx_channel, yy_channel = self.grid_32
        elif x_dim == 64:
            xx_channel, yy_channel = self.grid_64
        else:
            xx_channel, yy_channel = self.grid_128

        xx_channel, yy_channel = xx_channel.to(input_tensor.device), yy_channel.to(input_tensor.device)

        ret = torch.cat([
            input_tensor,
            xx_channel.repeat(batch_size, 1, 1, 1),
            yy_channel.repeat(batch_size, 1, 1, 1),
        ], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
        return ret


class CoordConv(nn.Module):

    def __init__(self, in_channels, out_channels, with_r=False, **kwargs):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_size = in_channels+2
        if with_r:
            in_size += 1
        self.conv = nn.Conv2d(in_size, out_channels, **kwargs)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)
        return ret

class DiscriminatorBlock(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.res = CoordConv(dim, dim_out, kernel_size = 1, stride = 2)

        self.net = nn.Sequential(
            CoordConv(dim, dim_out, kernel_size = 3, padding = 1),
            nn.LeakyReLU(0.2, inplace=True),
            CoordConv(dim_out, dim_out, kernel_size = 3, padding = 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.down = nn.AvgPool2d(2)

    def forward(self, x):
        res = self.res(x)
        x = self.net(x)
        x = self.down(x)
        x = x + res
        return x


class CoordPG2(nn.Module):
    def __init__(self, in_dim=3, n_feat=512, img_size=64, resolution0_pg=32, 
                 scale_img_domain=True, pg_milestones=[20000, 80000], 
                 fade_time=15000, double_fade_for_final_stage=False, **kwargs):
        super().__init__()

        self.fade_time = fade_time
        self.double_fade_for_final_stage = double_fade_for_final_stage
        if double_fade_for_final_stage:
            print("#" * 100)
            print("Using double fade in for discriminator.")
        # eg 6 for res=128
        layers = int(np.log2(img_size)) - 1
        max_chan = 400
        init_chan = 64
        chans = list(reversed(list(map(lambda t: 2 ** (11 - t), range(layers)))))
        chans = list(map(lambda n: min(max_chan, n), chans))
        # [64, 128, 256, 400, 400, 400]
        final_chan = chans[-1]
        n_feat0 = chans[0]

        self.scale_img_domain = scale_img_domain
        self.pg_milestones = pg_milestones


        # main block
        net = []
        for i in range(layers - 1):
            filter_in = chans[i]
            filter_out = chans[i+1]
            net += [
                DiscriminatorBlock(filter_in, filter_out)
            ]
        self.net = nn.ModuleList(net)

        n_pg_mappings = int(np.log2(img_size) - np.log2(resolution0_pg) + 1)
        self.n_pg_mappings = nn.ModuleList([
            CoordConv(3, chans[i], kernel_size = 1) for i in range(n_pg_mappings)
        ])
        self.pg_dims = [int(resolution0_pg * (2 ** (i))) for i in range(n_pg_mappings)]
        self.pg_dims.reverse()

        self.conv_out = CoordConv(chans[-1], 1, kernel_size = 4)        
        self.actvn = nn.LeakyReLU(0.2, inplace=True)

    def get_pg_idx(self, it):
        idx = 2
        for ms in self.pg_milestones:
            if it > ms:
                idx -= 1
        return idx

    def get_blending_weight(self, it):
        # todo make nicer
        ms = self.pg_milestones
        assert(len(ms) == 2)
        if (it < ms[0]):
            return 1., 0.
        elif (it >=ms[0]) and (it <ms[1]):
            w1 = (it - ms[0]) / self.fade_time
            w1 = max(0, min(1, w1))
            return (1 - w1), w1
        else:
            if self.double_fade_for_final_stage:
                w1 = (it - ms[1]) / (self.fade_time * 2)
            else:
                w1 = (it - ms[1]) / self.fade_time
            w1 = max(0, min(w1, 1))
            return (1-w1), w1
        
    def downsample(self, x):
        res = x.shape[-1] // 2
        return F.interpolate(x, (res, res), mode='bilinear', align_corners=True)

    def get_it(self, it):
        if type(it) not in [float, int]:
            it = it.reshape(-1)[0]
        return it

    def forward(self, x, it=0, **kwargs):
        it = self.get_it(it)

        batch_size = x.shape[0]
        res = x.shape[-1]
        assert(res in self.pg_dims)
        if self.scale_img_domain:
            x = x * 2. - 1.
        
        # idx = self.pg_dims.index(res)
        idx = self.get_pg_idx(it)

        # 32 x 32
        if idx == 2:
            # First 1x1 mapping
            net = self.actvn(self.n_pg_mappings[idx](x))
            # net = self.actvn(conv0)
            start_block_idx = 2
        # 64 x 64
        elif idx == 1:
            conv0 = self.actvn(self.n_pg_mappings[idx+1](self.downsample(x)))
            conv1 = self.actvn(self.n_pg_mappings[idx](x))
            # conv1 = self.net[idx](self.actvn(conv1))
            conv1 = self.net[idx](conv1)
            w0, w1 = self.get_blending_weight(it)
            net = conv0 * w0 + conv1 * w1
            start_block_idx = 2
        elif idx == 0:
            conv0 = self.actvn(self.n_pg_mappings[idx+1](self.downsample(x)))
            conv1 = self.actvn(self.n_pg_mappings[idx](x))
            conv1 = self.net[idx](conv1)
            w0, w1 = self.get_blending_weight(it)
            net = conv0 * w0 + conv1 * w1
            start_block_idx = 1

        # block
        for conv in self.net[start_block_idx:]:
            net = conv(net)
        
        # Final output mapping
        out = self.conv_out(net)
        out = out.reshape(batch_size, 1)
        return out
