import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from torchvision.ops import sigmoid_focal_loss

from src.unet_parts import DoubleConv, Down, OutConv, Up


class UNetModelReg(L.LightningModule):
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
        self.up4 = Up(192, 96, self.bilinear)

        self.outc = OutConv(96, self.out_seq_len)
        self.loss = nn.BCEWithLogitsLoss(pos_weight=self.init_weight_class())

    def init_weight_class(
        self,
    ):
        weight_full = torch.ones(12 * 3, 252, 252)
        for i in range(12):
            weight_full[3 * i] *= self.weights_class[0]
            weight_full[3 * i + 1] *= self.weights_class[1]
            weight_full[3 * i + 2] *= self.weights_class[2]
        return weight_full

    def freeze_weights(self):
        for name, param in self.named_parameters():
            if name.find("up4") != -1 or name.find("outc") != -1:
                param.requires_grad = True
            else:
                param.requires_grad = False

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
        loss = self.compute_loss(out, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch):
        x, y = batch
        out = self.forward(x)
        focal_loss = self.compute_loss(out, y)
        metrics = {
            f"val_loss": focal_loss,
        }
        self.log_dict(metrics, prog_bar=True)
        return metrics

    # def compute_loss(self, out, y):
    #     out[y==-1] = 0
    #     y[y==-1] = 0
    #     loss = self.loss(out, y)
    #     return loss

    def compute_loss(self, out, y):
        out[y == -1] = 0
        y[y == -1] = 0
        loss = nn.functional.l1_loss(torch.sigmoid(out), y)
        return loss

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
