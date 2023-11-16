import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR

from src.unet_parts import DoubleConv, Down, OutConv, Up


class UNetModelReg(L.LightningModule):
    def __init__(self, num_iters=100, num_epoch=100):
        super().__init__()
        self.num_channels = 4
        self.num_iters = num_iters
        self.out_seq_len = 12
        self.bilinear = True
        self.num_epoch = num_epoch

        self.inc = DoubleConv(self.num_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, self.bilinear)
        self.up2 = Up(512, 256 // factor, self.bilinear)
        self.up3 = Up(256, 128 // factor, self.bilinear)
        self.up4 = Up(128, 64, self.bilinear)

        self.outc = OutConv(64, self.out_seq_len)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def training_step(self, batch):
        x, y = batch
        out = self.forward(x)
        out[y == -1] = -1
        loss = F.mse_loss(out, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch):
        x, y = batch
        out = self.forward(x)
        out[y == -1] = -1
        loss = F.mse_loss(out, y)

        metrics = {
            f"val_loss": loss,
        }
        self.log_dict(metrics, prog_bar=True)
        return metrics

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": OneCycleLR(
                    optimizer,
                    max_lr=3e-4,
                    steps_per_epoch=self.num_iters,
                    epochs=self.num_epoch,
                ),
                "interval": "step",
            },
        }
