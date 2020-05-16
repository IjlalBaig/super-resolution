import random

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import pytorch_lightning as ptl
from math import log2
from collections import OrderedDict

from collections import OrderedDict

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1].expanduser().absolute()))


import logging
logger = logging.getLogger(__file__)

from src.tools.dataset import SRDataset
from src.models.generator import ProSRGenerator
from src.models.discriminator import ProSRDiscriminator
from src.models.loss import GANLoss


class ProSR(ptl.LightningModule):
    def __init__(self, in_planes=3, pyramid_in_planes=32, growth_rate=2, bn_size=4, **opts):
        super(ProSR, self).__init__()
        # todo: pass are args after testing
        blocks_cfg = [[4, 4, 4, 4, 4, 4], [4, 4, 4], [4]]
        planes_cfg = [[64, 128, 256, 512], [64], [64]]
        # todo: end
        self.opts = opts
        self.num_pyramids = len(blocks_cfg)
        gen_params = dict(in_planes=in_planes,
                          planes=pyramid_in_planes,
                          blocks_cfg=blocks_cfg,
                          growth_rate=growth_rate,
                          bn_size=bn_size)
        self.generator = ProSRGenerator(**gen_params)

        disc_params = dict(in_planes=in_planes,
                           planes_cfg=planes_cfg)
        self.discriminator = ProSRDiscriminator(**disc_params)

    def forward(self, x, upscale_factor, disc_only=False):
        if disc_only:
            return self.discriminator(x, upscale_factor)
        else:
            gen_out = self.generator(x, upscale_factor)
            return gen_out, self.discriminator(gen_out, upscale_factor)

    def configure_optimizers(self):
        lr_gen = self.opts.get("lr_gen", 1e-3)
        optim_gen = torch.optim.Adam(self.generator.parameters(), lr=lr_gen)

        lr_disc = self.opts.get("lr_disc", 1e-3)
        optim_disc = torch.optim.Adam(self.discriminator.parameters(), lr=lr_disc)
        return [optim_gen, optim_disc]

    def train_dataloader(self):
        train_path = self.opts.get("train_path")
        batch_size = self.opts.get("batch_size")
        num_workers = self.opts.get("num_workers")
        return DataLoader(SRDataset(train_path), batch_size=batch_size, shuffle=True,
                          pin_memory=True, num_workers=num_workers)

    def training_step(self, batch, batch_idx, optimizer_idx):
        max_upscale_factor = 2 ** self.num_pyramids
        itr_pyramid_id = batch_idx % self.num_pyramids

        x_size = batch.size(-2) // max_upscale_factor + \
                 batch.size(-2) // max_upscale_factor % 2
        y = F.interpolate(batch, size=(x_size * 2 ** (itr_pyramid_id + 1),
                                       x_size * 2 ** (itr_pyramid_id + 1)),
                                            mode="bilinear", align_corners=False)
        x = F.interpolate(batch, size=(x_size, x_size),
                            mode="bilinear", align_corners=False)

        upscale_factor = 2 ** (itr_pyramid_id + 1)
        if optimizer_idx == 0:
            gen_out, fake_labels = self(x, upscale_factor)
            g_loss = fake_labels.mean()

            tqdm_dict = {"g_loss": g_loss}
            output = OrderedDict({
                "loss": g_loss,
                "progress_bar": tqdm_dict,
                "log": tqdm_dict
            })
            return output

        if optimizer_idx == 1:
            gen_out, fake_labels = self(x, upscale_factor)
            real_labels = self(y, upscale_factor, disc_only=True)
            d_loss = (real_labels - fake_labels).mean()

            tqdm_dict = {"d_loss": d_loss}
            output = OrderedDict({
                "loss": d_loss,
                "progress_bar": tqdm_dict,
                "log": tqdm_dict
            })
            return output

    def optimizer_step( self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure):

        if optimizer_idx == 0:
            optimizer.step()
            optimizer.zero_grad()

        if optimizer_idx == 1:
            if batch_idx % self.opts.get("gen_steps_per_update", 10) == 0:
                optimizer.step()
                optimizer.zero_grad()
        pass

    def val_dataloader(self):
        val_path = self.opts.get("val_path")
        batch_size = self.opts.get("batch_size")
        num_workers = self.opts.get("num_workers")
        return DataLoader(SRDataset(val_path), batch_size=batch_size, shuffle=False,
                          pin_memory=True, num_workers=num_workers)

    def log_images(self, x, y, pred):
        self.logger.experiment.add_image("input", make_grid(x), self.current_epoch)
        self.logger.experiment.add_image("pred", make_grid(pred), self.current_epoch)
        self.logger.experiment.add_image("true", make_grid(y), self.current_epoch)

    def validation_step(self, batch, batch_idx):
        max_upscale_factor = 2 ** self.num_pyramids
        itr_pyramid_id = batch_idx % self.num_pyramids

        x_size = batch.size(-2) // max_upscale_factor + \
                 batch.size(-2) // max_upscale_factor % 2
        y = F.interpolate(batch, size=(x_size * 2 ** (itr_pyramid_id + 1),
                                       x_size * 2 ** (itr_pyramid_id + 1)),
                                        mode="bilinear", align_corners=False)
        x = F.interpolate(batch, size=(x_size, x_size),
                                    mode="bilinear", align_corners=False)

        upscale_factor = 2 ** (itr_pyramid_id + 1)

        gen_out, fake_labels = self(x, upscale_factor)
        real_labels = self(y, upscale_factor, disc_only=True)
        g_loss = fake_labels.mean()
        d_loss = (real_labels - fake_labels).mean()

        if batch_idx == 0:
            self.log_images(x, y, gen_out)

        output = OrderedDict({
            "g_loss": g_loss,
            "d_loss": d_loss,
        })
        return output

    def validation_epoch_end(self, outs):
        avg_g_loss = torch.stack([x["g_loss"] for x in outs]).mean()
        avg_d_loss = torch.stack([x["d_loss"] for x in outs]).mean()
        tqdm_dict = {"d_loss": avg_d_loss, "g_loss": avg_g_loss}
        return {"avg_g_loss": avg_g_loss,
                "avg_d_loss": avg_d_loss,
                "progress_bar": tqdm_dict,
                "log": tqdm_dict}


