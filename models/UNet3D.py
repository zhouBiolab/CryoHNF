#3D Unet with pos embedding

import math
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        """
        :param noise_level: B*C
        :return:
        """
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(1) \
                   * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding

class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels*(1+self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1,1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1,1)
        return x

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Upsample(nn.Module):
    def __init__(self, dim):
        #dim specifies the chanels
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        #nn.Conv3d(filters[0], self.CatChannels, 3, padding=1)
        self.conv = nn.Conv3d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))

class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv3d(dim, dim, 3, padding=1)
        self.down = nn.MaxPool3d(kernel_size=2)
    def forward(self, x, maxpool=True):
        if maxpool is True:
            return self.conv(self.down(x))
        else:
            return self.conv(x)

class Block(nn.Module):
    def __init__(self, dim, dim_out=1, groups=32, dropout=0):
        #dim specifies channels
        #dim_out specifies channels
        super().__init__()
        self.block = nn.Sequential(
            # nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv3d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out,
                 noise_level_emb_dim=None, dropout=0,
                 use_affine_level=False, norm_groups=32):
        super().__init__()
        if noise_level_emb_dim is not None:
            self.noise_func = FeatureWiseAffine(
            noise_level_emb_dim, dim_out, use_affine_level)
        else:
            self.noise_func =None
        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        b, c, h, w, l = x.shape
        h = self.block1(x)
        if time_emb is not None and self.noise_func is not None:
            h = self.noise_func(h, time_emb)
        h = self.block2(h)
        return h + self.res_conv(x)

def half_linear(input, weight, bias):
    # print('half', input.shape, weight.shape)
    if bias is not None:
        return F.linear(input, weight.half(), bias.half())
    else:
        return F.linear(input, weight.half())


def single_linear(input, weight, bias):
    # print('single', input.shape, weight.shape)
    # assert input.shape[0] < 10000

    return F.linear(input, weight, bias)
class MyLinear(nn.Linear):
    def forward(self, input):
        if input.dtype == torch.half:
            return half_linear(
                input, self.weight, self.bias
            )  # F.linear(input, self.weight.half(), self.bias.half())
        else:
            return single_linear(
                input, self.weight, self.bias
            )  # F.linear(input, self.weight, self.bias)
class ResidLinear(nn.Module):
    def __init__(self, nin, nout, bias = True):
        super(ResidLinear, self).__init__()
        self.linear = MyLinear(nin, nout, bias = bias)
        # self.linear = nn.utils.weight_norm(MyLinear(nin, nout))

    def forward(self, x):
        z = self.linear(x) + x
        return z

class ResidLinearMLP(nn.Module):
    def __init__(self, in_dim, nlayers, hidden_dim, out_dim, activation, bias = True):
        super(ResidLinearMLP, self).__init__()
        if in_dim == 512:
            layers = [MyLinear(in_dim, in_dim//2, bias=bias), activation(inplace=True),]
            layers.append(MyLinear(in_dim//2, in_dim//4, bias = bias))
        if in_dim == 64:
            layers = [MyLinear(in_dim, in_dim*2, bias=bias), activation(inplace=True), ]
        if in_dim == 8:
            layers = [MyLinear(in_dim, in_dim * 16, bias=bias), activation(inplace=True), ]
        for _ in range(nlayers):
            layers.append(ResidLinear(hidden_dim, hidden_dim, bias = bias))
            layers.append(activation(inplace=True))
        layers.append(
            ResidLinear(hidden_dim, out_dim, bias = bias)
            if out_dim == hidden_dim
            else MyLinear(hidden_dim, out_dim, bias = bias)
        )
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        # x = B x 1 x D x D x D
        x = x.view(x.size(0), -1)
        x = self.main(x)# x = B x 1
        return x

class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv3d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv3d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width, length = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width,length)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchwl, bncyxz -> bnhwlyxz", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, length, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, length, height, width,length)

        out = torch.einsum("bnhwlyxz, bncyxz -> bnchwl", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width,length))

        return out + input

class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, noise_level_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups)

    def forward(self, x, z_emb):
        x = self.res_block(x, z_emb)
        if(self.with_attn):
            x = self.attn(x)
        return x

class UNet3D(nn.Module):
    def __init__(
        self,
        activation,
        in_channel=1,
        out_channel=1,
        inner_channel=16,
        norm_groups=16,
        channel_mults=(1, 2, 4, 4),
        #channel_mults=(1, 2, 4),
        attn_res=[4],
        z_dim=0,
        res_blocks=3,
        dropout=0,
        with_noise_level_emb=False,
        box_size=24,
    ):
        super().__init__()
        if with_noise_level_emb:
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None
        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = box_size
        downs = [nn.Conv3d(in_channel, inner_channel,
                           kernel_size=3, padding=1)]

        self.mlp = nn.Sequential(
            nn.Linear(in_channel+z_dim+24, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 8)
        )

        #configure downsampling layers
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)
            channel_mult = int(inner_channel * channel_mults[ind])
            #continue double channels
            for _ in range(0, res_blocks):
                downs.append(ResnetBlocWithAttn(
                    pre_channel+z_dim, channel_mult, noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups, dropout=dropout, with_attn=use_attn))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res//2

        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel+z_dim, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                               dropout=dropout, with_attn=True),
            ResnetBlocWithAttn(pre_channel+z_dim, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                               dropout=dropout, with_attn=False)
        ])

        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks+1):
                ups.append(ResnetBlocWithAttn(
                    pre_channel+feat_channels.pop()+z_dim, channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                        dropout=dropout, with_attn=use_attn))
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res*2

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)
        # self.ResidLinearMLP = ResidLinearMLP(box_size**3, 0, 128, 1, activation)

    def process_layer(self, layer, x, t=None, z=None):
        if z is not None and not isinstance(layer, Downsample) and not isinstance(layer, Upsample) and not isinstance(layer, nn.Conv3d):
            x = torch.cat((x, z.expand(-1, -1, x.shape[-1], x.shape[-1], x.shape[-1])), dim=1)
        return layer(x, t) if isinstance(layer, ResnetBlocWithAttn) else layer(x)

    def forward(self, x, z=None, t=None):
        #time here is the gamma
        t = self.noise_level_mlp(t) if exists(
            self.noise_level_mlp) else None

        feats = []
        # x = self.mlp(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        for layer in self.downs:
            x = self.process_layer(layer, x, t, z)
            feats.append(x)

        for layer in self.mid:
            x = self.process_layer(layer, x, t, z)

        for layer in self.ups:
            #connections from downsampling and upsamling.
            if isinstance(layer, ResnetBlocWithAttn):
                x = torch.cat((x, feats.pop()), dim=1)
            x = self.process_layer(layer, x, t, z)
        x = self.process_layer(self.final_conv, x, t, None)
        # x = self.final_conv(x) if z is None else self.final_conv(torch.cat((x, z), dim=1)) # x = B x 1 x box_size x box_size x box_size
        return x
        # return self.ResidLinearMLP(x) # B x 1
