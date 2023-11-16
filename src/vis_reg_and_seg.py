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
from val_test import evaluate_on_val, process_test

from src.augmentation import init_aug
from src.config_trainer import init_trainer
from src.datasets import ProcessedRadarDataset, RadarDataset
from src.models import ConvLSTMModel, PersistantModel
from src.unet_classify import UNetModelClassify
from src.unet_multi_label import UNetModelMulti
from src.unet_reg import UNetModelReg


def prepare_data_loaders(train_batch_size=1, valid_batch_size=1, test_batch_size=1):
    valid_dataset = ProcessedRadarDataset(
        file_path="val.hdf5", preparation="regression", target_source=True
    )

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


def evaluate_on_val(seg, reg, valid_loader):
    outdir = "./plt_vis/seg_reg_" + str(datetime.now())
    os.makedirs(outdir, exist_ok=True)
    iter_dt = iter(valid_loader)
    for i in range(1):
        inputs, target = next(iter_dt)
    target = np.array(target)

    out_seg = torch.sigmoid(seg(inputs.to("cuda"))).detach().cpu().numpy()
    trsh = 0.5
    for i in range(12):
        out_seg[:, 3 * i][out_seg[:, 3 * i] < trsh] = 0
        out_seg[:, 3 * i + 1][
            (out_seg[:, 3 * i] < trsh) | (out_seg[:, 3 * i + 1] < trsh)
        ] = 0
        out_seg[:, 3 * i + 2][
            (out_seg[:, 3 * i + 1] < trsh) | (out_seg[:, 3 * i + 2] < trsh)
        ] = 0

        out_seg[:, 3 * i + 2][out_seg[:, 3 * i + 2] > trsh] = 1
        out_seg[:, 3 * i + 1][
            (out_seg[:, 3 * i + 1] > trsh) & (out_seg[:, 3 * i + 2] < trsh)
        ] = 1
        out_seg[:, 3 * i][
            (out_seg[:, 3 * i] > trsh) & (out_seg[:, 3 * i + 1] < trsh)
        ] = 1

    out_reg = torch.sigmoid(reg(inputs.to("cuda"))).detach().cpu().numpy()
    out_reg[out_seg != 1] = 0

    target = np.squeeze(target, axis=0)
    out_reg = np.squeeze(out_reg, axis=0)
    out_seg = np.squeeze(out_seg, axis=0)
    output = inverse(out_reg)

    output[target == -1] = -1
    for idx in range(12):
        plt.imsave(os.path.join(outdir, "target_" + str(idx) + ".png"), target[idx])
        plt.imsave(os.path.join(outdir, "predict_" + str(idx) + ".png"), output[idx])
        plt.imsave(
            os.path.join(outdir, "seg_" + str(idx) + ".png"),
            out_seg[3 * idx] + out_seg[3 * idx + 1] + out_seg[3 * idx + 2],
        )

    # return np.mean(np.sqrt(rmses))


def main():
    valid_loader = prepare_data_loaders()
    checkpoint_reg = "checkpoint/2023-11-09 01:33:20.721233/unet_reg_tune_epoch=64_val_loss=0.253680.ckpt"
    checkpoint_seg = "checkpoint/2023-11-06 18:47:51.750923/unet_multi_iter_epoch=23_val_loss=0.637777.ckpt"
    reg = UNetModelMulti()
    reg.load_state_dict(torch.load(checkpoint_reg)["state_dict"])
    reg.eval()
    reg.to("cuda")
    seg = UNetModelMulti()
    seg.load_state_dict(torch.load(checkpoint_seg)["state_dict"])
    seg.eval()
    seg.to("cuda")
    print(evaluate_on_val(seg, reg, valid_loader))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="unet_multi")
    parser.add_argument(
        "--checkpoint",  # default='checkpoint/2023-11-06 18:49:10.504195/unet_multi_epoch=23_val_loss=0.243418.ckpt',)
        # default='checkpoint/2023-11-06 18:47:51.750923/unet_multi_iter_epoch=23_val_loss=0.637777.ckpt')
        # default='checkpoint/2023-11-05 22:29:57.574119/unet_multi_epoch=47_val_loss=0.561606.ckpt')
        default="checkpoint/2023-11-09 01:33:20.721233/unet_reg_tune_epoch=64_val_loss=0.253680.ckpt",
    )
    args = parser.parse_args()
    main()
