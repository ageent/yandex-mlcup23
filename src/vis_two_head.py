import argparse
import os
from datetime import datetime

import h5py
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data
import tqdm

from src.augmentation import init_aug
from src.config_trainer import init_trainer
from src.datasets import IntensityRegSegDataset, ProcessedRadarDataset
from src.models import ConvLSTMModel, PersistantModel
from src.unet_classify import UNetModelClassify
from src.unet_multi_label import UNetModelMulti
from src.unet_reg import UNetModelReg
from src.unet_two_head import UNetModelTwoHead


def prepare_data_loaders(train_batch_size=1, valid_batch_size=1, test_batch_size=1):
    valid_dataset = IntensityRegSegDataset(file_path="val.hdf5", target_source=True)

    valid_loader = data.DataLoader(
        valid_dataset,
        batch_size=valid_batch_size,
        num_workers=10,
        pin_memory=True,
        shuffle=False,
    )

    return valid_loader


def inverse_channel(data, min, max):
    mask = data < 0.01
    data = data * (max - min) + min
    data[mask] = 0
    return data


def inverse(data):
    new_data = np.zeros((12, data.shape[1], data.shape[2]))
    for idx in range(12):
        new_data[idx] = (
            inverse_channel(data[3 * idx], 0, 1)
            + inverse_channel(data[3 * idx + 1], 1, 4)
            + inverse_channel(data[3 * idx + 2], 4, 50)
        )
    return new_data


def stack_res(bin_seg, reg, channels=3, to_abs=False):
    _, h, w = bin_seg.shape

    bin_seg = bin_seg.clone().view(-1, channels, h, w)
    masked_reg = reg.clone().view(-1, channels, h, w)

    ones_mask0 = bin_seg[:, 0] & bin_seg[:, 1]
    ones_mask1 = ones_mask0 & bin_seg[:, 2]
    ones_mask = torch.stack([ones_mask0, ones_mask1], dim=1)

    reg_mask0 = bin_seg[:, 0] & (~bin_seg[:, 1])
    reg_mask1 = ones_mask0 & (~bin_seg[:, 2])
    reg_mask2 = ones_mask1 & bin_seg[:, -1]
    reg_mask = torch.stack([reg_mask0, reg_mask1, reg_mask2], dim=1)

    masked_reg[~reg_mask] = 0
    masked_reg[:, :-1] = masked_reg[:, :-1] + ones_mask

    if to_abs:
        scale = torch.Tensor([1, 3, 46]).view(1, -1, 1, 1)
        masked_reg = masked_reg.mul_(scale)

    return masked_reg.sum(dim=1)


def evaluate_on_val(model, valid_loader):
    outdir = "./plt_vis/two_head" + str(datetime.now())
    os.makedirs(outdir, exist_ok=True)
    iter_dt = iter(valid_loader)
    print(len(valid_loader))
    for i in range(4000):
        inputs, target = next(iter_dt)
        # inputs, target = next(iter_dt)
    out_seg, out_reg = model(inputs.to("cuda"))
    out_seg = torch.sigmoid(out_seg).detach().cpu()
    out_reg = torch.sigmoid(out_reg).detach().cpu()

    target = torch.squeeze(target)
    out_reg = torch.squeeze(out_reg)
    out_seg = torch.squeeze(out_seg)
    out_seg[2::3] = 0
    trsh = 0.5
    out_seg = out_seg > trsh
    out_reg[out_reg < 0.01] = 0
    output = stack_res(out_seg, out_reg, to_abs=True)

    output = np.array(output)
    out_seg = np.array(out_seg)
    target = np.array(target)

    for idx in range(12):
        v = np.linspace(-0.1, 2.0, 3, endpoint=True)
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        im = ax[0].imshow(out_seg[3 * idx])
        fig.colorbar(im, ax=ax[0], ticks=v)

        im = ax[1].imshow(out_seg[3 * idx + 1])
        fig.colorbar(im, ax=ax[1], ticks=v)

        im = ax[2].imshow(out_seg[3 * idx + 2])
        fig.colorbar(im, ax=ax[2], ticks=v)

        plt.savefig(os.path.join(outdir, "trsh_seg_" + str(idx) + ".png"))

    output[target == -1] = -1
    print(output.shape)
    max_value = max(output.max(), target.max())
    for idx in range(12):
        target[idx, 0, 0] = max_value
        output[idx, 0, 0] = max_value
        fig, ax = plt.subplots(1, 2, figsize=(18, 6))
        im = ax[0].imshow(target[idx])

        fig.colorbar(im, ax=ax[0])

        im = ax[1].imshow(output[idx])
        fig.colorbar(im, ax=ax[1])

        plt.savefig(os.path.join(outdir, str(idx) + ".png"))


def main():
    valid_loader = prepare_data_loaders()
    checkpoint = "checkpoint/2023-11-10 00:04:34.487578/unet_two_head_epoch=54_val_loss=0.132021.ckpt"
    model = UNetModelTwoHead()
    model.load_state_dict(torch.load(checkpoint)["state_dict"])
    model.eval()
    model.to("cuda")
    print(evaluate_on_val(model, valid_loader))


if __name__ == "__main__":
    main()
