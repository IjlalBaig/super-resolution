from torch import nn
import torch.nn.functional as F
from math import log2

from src.models.components import conv3x3, PyramidDownBlock

import logging
logger = logging.getLogger(__file__)


class ProSRDiscriminator(nn.Module):
    def __init__(self, in_planes, planes_cfg):
        super(ProSRDiscriminator, self).__init__()

        self.num_pyramids = len(planes_cfg)
        for i in range(self.num_pyramids):
            plane_cfg = planes_cfg[i]
            self.add_module("init_conv_%d" % i, conv3x3(in_planes, plane_cfg[0]))

            self.add_module("pyramid_%d" % i, PyramidDownBlock(in_planes=plane_cfg[0],
                                                               downplanes_config=plane_cfg))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, upscale_factor=None):
        upscale_factor = self.get_valid_upscalefactor(upscale_factor)
        num_pyramids = int(log2(upscale_factor))

        for i in range(num_pyramids)[::-1]:
            if i == num_pyramids - 1:
                out = self.interpolate(x, scale_factor=1 / 2 ** (num_pyramids - i - 1))
                out = getattr(self, "init_conv_%d" % i)(out)
            out = getattr(self, "pyramid_%d" % i)(out)
        return self.sigmoid(out)

    def interpolate(self, x, scale_factor):
        return F.interpolate(x, scale_factor=scale_factor, mode="bicubic", align_corners=True).clamp(0, 1)

    def get_valid_upscalefactor(self, upscale_factor):
        if upscale_factor is None:
            upscale_factor = self.num_pyramids * 2
        else:
            valid_upscale_factors = [
                2 ** (i + 1) for i in range(self.num_pyramids)
            ]
            if upscale_factor not in valid_upscale_factors:
                logger.error("Invalid upscaling factor %s: choose one of: %s" % (
                    upscale_factor, valid_upscale_factors))
                raise SystemExit(1)
        return upscale_factor




