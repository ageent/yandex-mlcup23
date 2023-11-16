import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR

from src.unet_parts import DoubleConv, Down, OutConv, Up


class UNetModelClassify(L.LightningModule):
    def __init__(self, num_iters=100, num_epoch=100):
        super().__init__()
        self.num_channels = 4 * 4
        self.num_iters = num_iters
        self.out_seq_len = 12 * 4
        self.bilinear = True
        self.num_epoch = num_epoch
        self.weights_class = torch.Tensor([0.00246, 0.028, 0.2248, 0.74464]).to("cuda")
        self.inc = DoubleConv(self.num_channels, 96)
        self.down1 = Down(96, 192)
        self.down2 = Down(192, 384)
        self.down3 = Down(384, 768)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(768, 1536 // factor)
        self.up1 = Up(1536, 768 // factor, self.bilinear)
        self.up2 = Up(768, 384 // factor, self.bilinear)
        self.up3 = Up(384, 192 // factor, self.bilinear)
        self.up4 = Up(192, 96, self.bilinear)

        self.outc = OutConv(96, self.out_seq_len)

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
        loss = 0
        for i in range(y.shape[1]):
            loss += F.cross_entropy(
                out[:, 4 * i : 4 * i + 4],
                y[:, i].long(),
                weight=self.weights_class,
                ignore_index=-1,
            )
        loss = loss / y.shape[1]
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch):
        x, y = batch
        out = self.forward(x)
        loss = 0
        for i in range(y.shape[1]):
            loss += F.cross_entropy(
                out[:, 4 * i : 4 * i + 4],
                y[:, i].long(),
                weight=self.weights_class,
                ignore_index=-1,
            )
        loss = loss / y.shape[1]

        metrics = {
            f"val_loss": loss,
        }
        self.log_dict(metrics, prog_bar=True)
        return metrics

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), weight_decay=1e-4)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": OneCycleLR(
                    optimizer,
                    max_lr=3e-5,
                    steps_per_epoch=self.num_iters,
                    epochs=self.num_epoch,
                ),
                "interval": "step",
            },
        }
