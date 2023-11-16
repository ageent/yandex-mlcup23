import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from torchvision.ops import sigmoid_focal_loss

from src.unet_parts import DoubleConv, Down, OutConv, Up


class UNetModelTwoHead(L.LightningModule):
    def __init__(self, num_iters=100, num_epoch=100):
        super().__init__()
        self.num_channels = 4 * 3
        self.num_iters = num_iters
        self.out_seq_len = 12 * 3
        self.bilinear = True
        self.num_epoch = num_epoch
        self.weights_class = [6.943106588523985, 6.617503376086996, 6.178742411268663]

        self.inc = DoubleConv(self.num_channels, 96)
        self.down1 = Down(96, 192)
        self.down2 = Down(192, 384)
        self.down3 = Down(384, 768)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(768, 1536 // factor)

        self.up1 = Up(1536, 768 // factor, self.bilinear)
        self.up2 = Up(768, 384 // factor, self.bilinear)
        self.up3 = Up(384, 192 // factor, self.bilinear)
        self.h1_up4 = Up(192, 96, self.bilinear)
        self.h1_outc = OutConv(96, self.out_seq_len)

        self.h2_up4 = Up(192, 96, self.bilinear)
        self.h2_outc = OutConv(96, self.out_seq_len)

        self.loss = nn.BCEWithLogitsLoss(
            pos_weight=self.init_weight_class(), reduction="none"
        )

    def init_weight_class(
        self,
    ):
        weight_full = torch.ones(12 * 3, 252, 252)
        for i in range(12):
            weight_full[3 * i] *= self.weights_class[0]
            weight_full[3 * i + 1] *= self.weights_class[1]
            weight_full[3 * i + 2] *= self.weights_class[2]
        return weight_full

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x_h1 = self.h1_up4(x, x1)
        x_h2 = self.h2_up4(x, x1)
        logits_h1 = self.h1_outc(x_h1)
        logits_h2 = self.h2_outc(x_h2)
        return logits_h1, logits_h2

    def training_step(self, batch):
        x, (y_seg, y_reg) = batch
        out_seg, out_reg = self.forward(x)
        loss_BCE, loss_L1 = self.compute_loss(out_seg, out_reg, y_seg, y_reg)
        metrics = {
            f"train_loss": loss_BCE,
            f"train_loss_L1": loss_L1,
        }
        self.log_dict(metrics, prog_bar=True)
        return loss_BCE + loss_L1

    def validation_step(self, batch):
        x, (y_seg, y_reg) = batch
        out_seg, out_reg = self.forward(x)
        loss_BCE, loss_L1 = self.compute_loss(out_seg, out_reg, y_seg, y_reg)
        metrics = {
            f"val_loss": loss_BCE,
            f"val_loss_L1": loss_L1,
        }
        self.log_dict(metrics, prog_bar=True)
        return metrics

    def compute_loss(self, out_seg, out_reg, y_seg, y_reg):
        masked = lambda x, y: (x * (y != -1).float()).nanmean()

        out_reg = torch.sigmoid(out_reg)
        loss_L1 = masked(nn.functional.l1_loss(out_reg, y_reg, reduction="none"), y_reg)
        loss_BCE = masked(self.loss(out_seg, y_seg), y_seg)

        return loss_BCE, loss_L1

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
                    pct_start=0.05,
                ),
                "interval": "step",
            },
        }
