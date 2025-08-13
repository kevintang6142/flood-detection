import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from models.laplace import EA  # <-- Import the EA module


# =====================================================================================
# This is the custom DOConv2d layer.
# =====================================================================================
class DOConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        use_bias=False,
    ):
        super(DOConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        # This layer has two weight parameters, W and D.
        self.W = nn.Parameter(torch.randn(out_channels, in_channels, *self.kernel_size))
        self.D = nn.Parameter(torch.randn(out_channels, in_channels, *self.kernel_size))
        if use_bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

    def forward(self, x):
        # For inference, the loaded weights define the behavior.
        effective_kernel = self.W + self.D
        return F.conv2d(
            x, effective_kernel, self.bias, self.stride, self.padding, self.dilation
        )


# =====================================================================================
# The BasicConv wrapper correctly chooses which convolution type to use.
# =====================================================================================
class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias=False,
    ):
        super(BasicConv, self).__init__()
        # Use standard nn.Conv2d for 1x1 kernels
        if kernel_size == 1:
            self.conv = nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=bias,
            )
        # Use our custom DOConv2d for all others
        else:
            self.conv = DOConv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                use_bias=bias,
            )
        self.bn = (
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            if bn
            else None
        )
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


# =====================================================================================
# Definitive MDPM Module: Includes the EA submodule as required by the checkpoint.
# =====================================================================================
class MDPM(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        stride=1,
        scale=0.1,
        map_reduce=8,
        vision=1,
        groups=1,
    ):
        super(MDPM, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        self.inter_planes = in_planes // map_reduce

        # The branch structure and channel numbers are corrected to match the error log.
        self.branch0 = nn.Sequential(
            BasicConv(
                in_planes,
                self.inter_planes,
                kernel_size=1,
                stride=1,
                groups=groups,
                relu=False,
            ),
            BasicConv(
                self.inter_planes,
                8,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=groups,
            ),
            BasicConv(
                8,
                8,
                kernel_size=3,
                stride=1,
                padding=vision + 1,
                dilation=vision + 1,
                relu=False,
                groups=groups,
            ),
        )
        self.branch1 = nn.Sequential(
            BasicConv(
                in_planes,
                self.inter_planes,
                kernel_size=1,
                stride=1,
                groups=groups,
                relu=False,
            ),
            BasicConv(
                self.inter_planes,
                8,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=groups,
            ),
            BasicConv(
                8,
                8,
                kernel_size=3,
                stride=1,
                padding=vision + 2,
                dilation=vision + 2,
                relu=False,
                groups=groups,
            ),
        )
        self.branch2 = nn.Sequential(
            BasicConv(
                in_planes,
                self.inter_planes,
                kernel_size=1,
                stride=1,
                groups=groups,
                relu=False,
            ),
            BasicConv(
                self.inter_planes,
                6,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=groups,
            ),
            BasicConv(
                6,
                8,
                kernel_size=3,
                stride=1,
                padding=vision + 4,
                dilation=vision + 4,
                relu=False,
                groups=groups,
            ),
            BasicConv(
                8, 8, kernel_size=3, stride=1, padding=1, groups=groups, relu=False
            ),
        )  # Extra layer from checkpoint

        self.ConvLinear = BasicConv(
            8 + 8 + 8, out_planes, kernel_size=1, stride=1, relu=False
        )
        self.shortcut = BasicConv(
            in_planes, out_planes, kernel_size=1, stride=stride, relu=False
        )
        self.relu = nn.ReLU(inplace=False)

        # The EA submodule is restored here.
        self.EA = EA(in_planes)

    def forward(self, x, edge):
        # The EA module is now used in the forward pass as intended.
        x, am = self.EA(x, edge)
        short = self.shortcut(x)
        out = self.relu(short)
        return out, am
