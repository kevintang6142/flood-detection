# In FUSE.py
from collections import OrderedDict

import torch.nn as nn


class DFIM(nn.Module):
    def __init__(self, planes_high, planes_low, planes_out):
        super(DFIM, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(planes_low, planes_low // 4, 1),
            nn.BatchNorm2d(planes_low // 4),
            nn.ReLU(True),
            nn.Conv2d(planes_low // 4, planes_low, 1),
            nn.BatchNorm2d(planes_low),
            nn.Sigmoid(),
        )
        self.plus_conv = nn.Sequential(
            nn.Conv2d(planes_high, planes_low, 1),
            nn.BatchNorm2d(planes_low),
            nn.ReLU(True),
        )
        # This OrderedDict structure must match what the checkpoint expects.
        self.ca = nn.Sequential(
            OrderedDict(
                [
                    ("0", nn.AdaptiveAvgPool2d(1)),
                    ("1", nn.Conv2d(planes_low, planes_low // 4, 1)),
                    ("2", nn.ReLU(True)),
                    ("3", nn.Conv2d(planes_low // 4, planes_low, 1)),
                    ("4", nn.Sigmoid()),
                ]
            )
        )
        self.end_conv = nn.Sequential(
            nn.Conv2d(planes_low, planes_out, 3, 1, 1),
            nn.BatchNorm2d(planes_out),
            nn.ReLU(True),
        )

    def forward(self, x_high, x_low):
        x_high = self.plus_conv(x_high)
        pa = self.pa(x_low)
        ca = self.ca(x_high)
        feat = x_low + x_high
        feat = self.end_conv(feat)
        feat = feat * ca
        feat = feat * pa
        return feat
