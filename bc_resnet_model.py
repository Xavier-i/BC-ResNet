import math

import torch
from torch import nn
import torch.nn.functional as F

from subspectral_norm import SubSpectralNorm

DROPOUT = 0.1


class Conv2dSamePadding(nn.Conv2d):
    """2D Convolutions with same padding
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True, name=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation, groups=groups,
                         bias=bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2
        self.name = name

    def forward(self, x):
        input_h, input_w = x.size()[2:]
        kernel_h, kernel_w = self.weight.size()[2:]
        stride_h, stride_w = self.stride
        output_h, output_w = math.ceil(input_h / stride_h), math.ceil(input_w / stride_w)
        pad_h = max((output_h - 1) * self.stride[0] + (kernel_h - 1) * self.dilation[0] + 1 - input_h, 0)
        pad_w = max((output_w - 1) * self.stride[1] + (kernel_w - 1) * self.dilation[1] + 1 - input_w, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class NormalBlock(nn.Module):
    def __init__(self, n_chan: int, *, dilation: int = 1, dropout: float = DROPOUT, use_subspectral: bool = True):
        super().__init__()
        norm_layer = SubSpectralNorm(n_chan, 5) if use_subspectral else nn.BatchNorm2d(n_chan)
        self.f2 = nn.Sequential(
            Conv2dSamePadding(in_channels=n_chan, out_channels=n_chan, kernel_size=(3, 1), groups=n_chan),
            norm_layer,
        )
        self.f1 = nn.Sequential(
            Conv2dSamePadding(in_channels=n_chan, out_channels=n_chan, kernel_size=(1, 3), groups=n_chan,
                              dilation=(1, dilation)),
            nn.BatchNorm2d(n_chan),
            nn.SiLU(),
            nn.Conv2d(n_chan, n_chan, kernel_size=1),
            nn.Dropout2d(dropout)
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        n_freq = x.shape[2]
        x1 = self.f2(x)

        x2 = torch.mean(x1, dim=2, keepdim=True)
        x2 = self.f1(x2)
        x2 = x2.repeat(1, 1, n_freq, 1)

        return self.activation(x + x1 + x2)


class TransitionBlock(nn.Module):
    def __init__(self, in_chan: int, out_chan: int, *, dilation: int = 1, stride: int = 1, dropout: float = DROPOUT,
                 use_subspectral: bool = True):
        super().__init__()

        if stride == 1:
            conv = Conv2dSamePadding(in_channels=out_chan, out_channels=out_chan, kernel_size=(3, 1), groups=out_chan)
        else:
            conv = nn.Conv2d(out_chan, out_chan, kernel_size=(3, 1), stride=(stride, 1), groups=out_chan,
                             padding=(1, 0))

        norm_layer = SubSpectralNorm(out_chan, 5) if use_subspectral else nn.BatchNorm2d(out_chan)
        self.f2 = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=(1, 1)),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(),
            conv,
            norm_layer,
        )

        self.f1 = nn.Sequential(
            Conv2dSamePadding(in_channels=out_chan, out_channels=out_chan, kernel_size=(1, 3), groups=out_chan,
                              dilation=(1, dilation)),
            nn.BatchNorm2d(out_chan),
            nn.SiLU(),
            nn.Conv2d(out_chan, out_chan, kernel_size=1),
            nn.Dropout2d(dropout)
        )

        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = self.f2(x)
        n_freq = x.shape[2]
        x1 = torch.mean(x, dim=2, keepdim=True)
        x1 = self.f1(x1)
        x1 = x1.repeat(1, 1, n_freq, 1)

        return self.activation(x + x1)


class BcResNetModel(nn.Module):
    def __init__(self, n_class: int = 35, *, scale: int = 1, dropout: float = DROPOUT, use_subspectral: bool = True):
        super().__init__()

        self.input_conv = nn.Conv2d(1, 16 * scale, kernel_size=(5, 5), stride=(2, 1), padding=2)

        self.t1 = TransitionBlock(16 * scale, 8 * scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n11 = NormalBlock(8 * scale, dropout=dropout, use_subspectral=use_subspectral)

        self.t2 = TransitionBlock(8 * scale, 12 * scale, dilation=2, stride=2, dropout=dropout,
                                  use_subspectral=use_subspectral)
        self.n21 = NormalBlock(12 * scale, dilation=2, dropout=dropout, use_subspectral=use_subspectral)

        self.t3 = TransitionBlock(12 * scale, 16 * scale, dilation=4, stride=2, dropout=dropout,
                                  use_subspectral=use_subspectral)
        self.n31 = NormalBlock(16 * scale, dilation=4, dropout=dropout, use_subspectral=use_subspectral)
        self.n32 = NormalBlock(16 * scale, dilation=4, dropout=dropout, use_subspectral=use_subspectral)
        self.n33 = NormalBlock(16 * scale, dilation=4, dropout=dropout, use_subspectral=use_subspectral)

        self.t4 = TransitionBlock(16 * scale, 20 * scale, dilation=8, dropout=dropout, use_subspectral=use_subspectral)
        self.n41 = NormalBlock(20 * scale, dilation=8, dropout=dropout, use_subspectral=use_subspectral)
        self.n42 = NormalBlock(20 * scale, dilation=8, dropout=dropout, use_subspectral=use_subspectral)
        self.n43 = NormalBlock(20 * scale, dilation=8, dropout=dropout, use_subspectral=use_subspectral)

        self.dw_conv = nn.Conv2d(20 * scale, 20 * scale, kernel_size=(5, 5), groups=20)
        self.onexone_conv = nn.Conv2d(20 * scale, 32 * scale, kernel_size=1)

        self.head_conv = nn.Conv2d(32 * scale, n_class, kernel_size=1)

    def forward(self, x: torch.Tensor):
        x = self.input_conv(x)
        x = self.t1(x)
        x = self.n11(x)

        x = self.t2(x)
        x = self.n21(x)

        x = self.t3(x)
        x = self.n31(x)
        x = self.n32(x)
        x = self.n33(x)

        x = self.t4(x)
        x = self.n41(x)
        x = self.n42(x)
        x = self.n43(x)

        x = self.dw_conv(x)
        x = self.onexone_conv(x)

        x = torch.mean(x, dim=3, keepdim=True)
        x = self.head_conv(x)

        x = x.squeeze()

        return F.log_softmax(x, dim=-1)
