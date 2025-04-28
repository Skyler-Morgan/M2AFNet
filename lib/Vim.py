
import torch.nn as nn
from einops import rearrange
import numbers
from mamba_ssm.modules.mamba_simple import Mamba
import torch
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class ChannelExchange(nn.Module):
    def __init__(self, p=2):
        super().__init__()
        self.p = p

    def forward(self, x1, x2):
        x1 = x1.permute(0, 2, 1)
        x2 = x2.permute(0, 2, 1)

        B, C, N = x1.shape
        exchange_mask = torch.arange(C) % self.p == 0
        exchange_mask = exchange_mask.unsqueeze(0).expand((B, -1))
        out_x1, out_x2 = torch.zeros_like(x1), torch.zeros_like(x2)
        out_x1[~exchange_mask, ...] = x1[~exchange_mask, ...]
        out_x2[~exchange_mask, ...] = x2[~exchange_mask, ...]
        out_x1[exchange_mask, ...] = x2[exchange_mask, ...]
        out_x2[exchange_mask, ...] = x1[exchange_mask, ...]
        out_x1 = out_x1.permute(0, 2, 1)
        out_x2 = out_x2.permute(0, 2, 1)
        return out_x1, out_x2
class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        if len(x.shape) == 4:
            h, w = x.shape[-2:]
            return to_4d(self.body(to_3d(x)), h, w)
        else:
            return self.body(x)
class MambaBlock(nn.Module):
    def __init__(self, dim):
        super(MambaBlock, self).__init__()
        self.encoder = Mamba(dim)
        self.norm = LayerNorm(dim, 'with_bias')
        self.conv33conv33conv11 = nn.Sequential(
            nn.Conv1d(in_channels=101, out_channels=101, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(101),
            nn.ReLU()
        )
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim //2),
            nn.GELU(),
            nn.Linear(dim //2, dim),
        )

        self.norm = nn.LayerNorm(dim)
    def forward(self, ipt):
        x = self.conv33conv33conv11(ipt)
        x = self.norm(x)
        x1 = self.encoder(x)
        return x1
class MambaBlock3D(nn.Module):
    def __init__(self, dim):
        super(MambaBlock3D, self).__init__()
        self.encoder = Mamba(dim)
        self.norm = LayerNorm(dim, 'with_bias')
        self.conv33conv33conv11 = nn.Sequential(
            nn.Conv1d(in_channels=401, out_channels=401, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(401),
        )
        self.norm = nn.LayerNorm(dim)
    def forward(self, ipt):  #torch.Size([56, 101, 1024])

        x = self.conv33conv33conv11(ipt)
        x = self.norm(x)
        x1 = self.encoder(x)
        return x1

class TokenSwapMamba(nn.Module):
    def __init__(self, dim):
        super(TokenSwapMamba, self).__init__()
        self.I1encoder = Mamba(dim, bimamba_type=None)
        self.I2encoder = Mamba(dim, bimamba_type=None)
        self.norm1 = LayerNorm(dim, 'with_bias')
        self.norm2 = LayerNorm(dim, 'with_bias')
        self.ChannelExchange = ChannelExchange(p=2)

    def forward(self, I1, I2
                , I1_residual, I2_residual):
        I1_residual = I1 + I1_residual
        I2_residual = I2 + I2_residual
        I1 = self.norm1(I1_residual)
        I2 = self.norm2(I2_residual)
        B, N, C = I1.shape

        I1_swap, I2_swap = self.ChannelExchange(I1, I2)

        I1_swap = self.I1encoder(I1_swap)
        I2_swap = self.I2encoder(I2_swap)
        return I1_swap, I2_swap, I1_residual, I2_residual


class M3(nn.Module):
    def __init__(self, dim):
        super(M3, self).__init__()
        self.multi_modal_mamba_block = Mamba(dim, bimamba_type="m3")
        self.norm1 = LayerNorm(dim, 'with_bias')
        self.norm2 = LayerNorm(dim, 'with_bias')
        self.norm3 = LayerNorm(dim, 'with_bias')

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, I1, fusion_resi, I2, fusion, test_h, test_w):
        fusion_resi = fusion + fusion_resi
        fusion = self.norm1(fusion_resi)
        I2 = self.norm2(I2)
        I1 = self.norm3(I1)

        global_f = self.multi_modal_mamba_block(self.norm1(fusion), extra_emb1=self.norm2(I2),
                                                extra_emb2=self.norm3(I1))

        B, HW, C = global_f.shape
        fusion = global_f.transpose(1, 2).view(B, C, test_h, test_w)
        fusion = (self.dwconv(fusion) + fusion).flatten(2).transpose(1, 2)
        return fusion, fusion_resi