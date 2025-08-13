import torch
import torch.nn.functional as F
from torch import nn


class Conv3Relu(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(Conv3Relu, self).__init__()
        self.extract = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                (3, 3),
                padding=(1, 1),
                stride=(stride, stride),
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),  # <-- Reverted to BatchNorm2d
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.extract(x)


# In aspp.py
# In aspp.py
class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        # The checkpoint expects a simple sequential block, not a nested 'branch' module.
        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        size = x.shape[-2:]
        # Access layers by index and reorder operations to prevent 1x1 error
        pooled = self.layers[0](x)
        conved = self.layers[1](pooled)
        upsampled = F.interpolate(
            conved, size=size, mode="bilinear", align_corners=False
        )
        normed = self.layers[2](upsampled)
        relued = self.layers[3](normed)
        return relued


class RCSA_ASPP(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(RCSA_ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),  # <-- Reverted
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(
                dim_in, dim_out, 3, 1, padding=2 * rate, dilation=2 * rate, bias=True
            ),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),  # <-- Reverted
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(
                dim_in, dim_out, 3, 1, padding=4 * rate, dilation=4 * rate, bias=True
            ),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),  # <-- Reverted
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(
                dim_in, dim_out, 3, 1, padding=8 * rate, dilation=8 * rate, bias=True
            ),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),  # <-- Reverted
            nn.ReLU(inplace=True),
        )
        self.branch5 = ASPPPooling(dim_in, dim_out)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),  # <-- Reverted
            nn.ReLU(inplace=True),
        )
        self.stage1_Conv2 = Conv3Relu(dim_in * 2, dim_in)
        self.stage2_Conv2 = Conv3Relu(dim_in * 2, dim_in)
        self.stage3_Conv2 = Conv3Relu(dim_in * 2, dim_in)
        self.stage4_Conv2 = Conv3Relu(dim_in * 2, dim_in)

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        global_feature = self.branch5(x)
        out45 = self.stage1_Conv2(torch.cat([out4, global_feature], 1))
        out345 = self.stage2_Conv2(torch.cat([out45, out3], 1))
        out2345 = self.stage3_Conv2(torch.cat([out345, out2], 1))
        out12345 = self.stage4_Conv2(torch.cat([out2345, out1], 1))
        result = out12345
        return result
