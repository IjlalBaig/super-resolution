from torch import nn
import torch.nn.functional as F
from math import log2

from src.models.components import conv3x3, PyramidUpBlock, Mish

import logging
logger = logging.getLogger(__file__)


class ProSRGenerator(nn.Module):
    def __init__(self, in_planes, planes, blocks_cfg, bn_size, growth_rate):
        super(ProSRGenerator, self).__init__()

        self.num_pyramids = len(blocks_cfg)

        self.add_module("init_conv", conv3x3(in_planes, planes))
        for i in range(self.num_pyramids):
            block_cfg = blocks_cfg[i]
            self.add_module("pyramid_%d" % i, PyramidUpBlock(in_planes=planes,
                                                             out_planes=planes,
                                                             denseblock_config=block_cfg,
                                                             bn_size=bn_size,
                                                             growth_rate=growth_rate))
            self.add_module("reconst_%d" % i, conv3x3(planes, in_planes))

            self.init_weights()

    def forward(self, x, upscale_factor=None):
        upscale_factor = self.get_valid_upscalefactor(upscale_factor)
        num_pyramids = int(log2(upscale_factor))

        out = getattr(self, 'init_conv')(x)
        for i in range(num_pyramids):
            out = getattr(self, "pyramid_%d" % i)(out)
            if i == num_pyramids - 1:
                out_hi = Mish()(getattr(self, "reconst_%d" % i)(out))
                out_lo = F.interpolate(x.detach(), scale_factor=2 ** (i + 1), mode="bicubic", align_corners=True).clamp(0, 1)
                out = out_hi * out_lo
        return F.hardtanh(out, 0, 1), out_hi

    def get_valid_upscalefactor(self, upscale_factor):
        if upscale_factor is None:
            upscale_factor = self.num_pyramids * 2
        else:
            valid_upscale_factors = [
                2 ** (i + 1) for i in range(self.num_pyramids)
            ]
            if upscale_factor not in valid_upscale_factors:
                logger.error("Invalid upscaling factor %s: choose one of: %s" %(
                    upscale_factor, valid_upscale_factors))
                raise SystemExit(1)
        return upscale_factor

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                m.bias.data.zero_()

