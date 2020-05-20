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
from src.models.loss import GANLoss, VGGLoss
from argparse import Namespace


class ProSR(ptl.LightningModule):
    def __init__(self, model_cfg, **opts):
        super(ProSR, self).__init__()
        self.h_params = Namespace(**model_cfg.h_params)
        self.opts = Namespace(**opts)
        G_cfg = Namespace(**model_cfg.G)
        D_cfg = Namespace(**model_cfg.D)

        num_up_pyramids = len(G_cfg.blocks_cfg)
        num_down_pyramids = len(D_cfg.planes_cfg)
        assert num_up_pyramids == num_down_pyramids, \
            "num_up_pyramids == num_down_pyramids"
        self.num_pyramids = num_up_pyramids

        self.G = ProSRGenerator(G_cfg)
        self.D = ProSRDiscriminator(D_cfg)
        self.vgg_loss = VGGLoss()

    def forward(self, x, upscale_factor, disc_only=False):
        if disc_only:
            return self.D(x, upscale_factor)
        else:
            gen_out, _ = self.G(x, upscale_factor)
            return gen_out, self.D(gen_out, upscale_factor), _

    def configure_optimizers(self):
        optim_G= torch.optim.Adam(self.G.parameters(), lr=self.h_params.G_lr)
        optim_D = torch.optim.Adam(self.D.parameters(), lr=self.h_params.D_lr)
        return [optim_G, optim_D]

    def train_dataloader(self):
        return DataLoader(SRDataset(self.opts.train_path),
                          batch_size=self.h_params.batch_size,
                          num_workers=self.opts.num_workers,
                          pin_memory=True,
                          shuffle=True)

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
            G_out, fake_labels, _ = self(x, upscale_factor)
            G_loss = ((fake_labels - 1) ** 2).mean()
            vgg_loss = self.vgg_loss(G_out, y)

            tqdm_dict = dict(vgg_loss=vgg_loss, G_loss=G_loss)
            output = OrderedDict({
                "loss": vgg_loss + G_loss,
                "progress_bar": tqdm_dict,
                "log": tqdm_dict
            })
            return output

        if optimizer_idx == 1:
            D_out, fake_labels, _ = self(x, upscale_factor)
            real_labels = self(y, upscale_factor, disc_only=True)
            D_loss = (fake_labels ** 2 + (real_labels - 1) ** 2).mean()

            tqdm_dict = dict(D_loss=D_loss)
            output = OrderedDict({
                "loss": D_loss,
                "progress_bar": tqdm_dict,
                "log": tqdm_dict
            })
            return output

    def optimizer_step( self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure):

        if optimizer_idx == 0:
            optimizer.step()
            optimizer.zero_grad()

        if optimizer_idx == 1:
            if batch_idx % self.h_params.gen_steps_per_update == 0:
                optimizer.step()
                optimizer.zero_grad()
        pass

    def val_dataloader(self):
        return DataLoader(SRDataset(self.opts.val_path),
                          batch_size=self.h_params.batch_size,
                          num_workers=self.opts.num_workers,
                          pin_memory=True,
                          shuffle=False)

    def log_images(self, x, y, pred, pred_hi):
        x = F.interpolate(x.detach(), scale_factor=2, mode="bicubic", align_corners=True).clamp(0, 1)
        self.logger.experiment.add_image("input", make_grid(x), self.current_epoch)
        self.logger.experiment.add_image("pred", make_grid(pred), self.current_epoch)
        self.logger.experiment.add_image("pred_hi", make_grid(pred_hi), self.current_epoch)
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

        G_out, fake_labels, _ = self(x, upscale_factor)
        real_labels = self(y, upscale_factor, disc_only=True)
        G_loss = ((fake_labels - 1) ** 2).mean()
        D_loss = (fake_labels ** 2 + (real_labels - 1) ** 2).mean()
        vgg_loss = self.vgg_loss(G_out, y)

        if batch_idx == 0:
            self.log_images(x, y, G_out, _)

        output = OrderedDict({
            "G_loss": G_loss,
            "D_loss": D_loss,
            "loss": vgg_loss
        })
        return output

    def validation_epoch_end(self, outs):
        avg_loss = torch.stack([x["loss"] for x in outs]).mean()
        tqdm_dict = {"avg_loss": avg_loss}
        output = OrderedDict({
            "val_loss": avg_loss,
            "progress_bar": tqdm_dict,
            "log": tqdm_dict})
        return output


