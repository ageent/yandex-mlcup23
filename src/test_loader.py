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
from src.datasets import (FullFeatRegSegDataLoader, FullFeatRegSegDataset,
                          IntensityRegSegDataset)
from src.models import ConvLSTMModel, PersistantModel
from src.unet_classify import UNetModelClassify
from src.unet_multi_label import UNetModelMulti
from src.unet_reg import UNetModelReg
from src.unet_two_head import UNetModelTwoHead


def prepare_data_loaders(train_batch_size=1, valid_batch_size=1, test_batch_size=1):
    valid_dataset = FullFeatRegSegDataset(file_path="val_full.hdf5", device="cpu")
    transforms = init_aug()
    # transforms = None
    valid_loader = FullFeatRegSegDataLoader(
        transforms,
        valid_dataset,
        batch_size=valid_batch_size,
        num_workers=10,
        shuffle=False,
    )

    return valid_loader


def evaluate_on_val(valid_loader):
    outdir = "./plt_vis/two_head" + str(datetime.now())
    os.makedirs(outdir, exist_ok=True)
    iter_dt = iter(valid_loader)
    print(len(valid_loader))
    for i in range(4000):
        inputs, target = next(iter_dt)
    for idx_inp, inp in enumerate(inputs):
        inp = np.squeeze(np.array(inp))
        size = int(inp.shape[0] / 4)
        for t in range(4):
            fig, ax = plt.subplots(1, size, figsize=(18, 6))
            if size == 1:
                for cnl in range(size):
                    im = ax.imshow(inp[size * t + cnl])
                    fig.colorbar(im, ax=ax)
            else:
                for cnl in range(size):
                    im = ax[cnl].imshow(inp[size * t + cnl])
                    fig.colorbar(im, ax=ax[cnl])

            plt.savefig(os.path.join(outdir, f"inp_{idx_inp}_{t}" + ".png"))

    for idx_inp, inp in enumerate(target):
        inp = np.squeeze(np.array(inp))
        size = int(inp.shape[0] / 12)
        for t in range(12):
            fig, ax = plt.subplots(1, size, figsize=(18, 6))
            for cnl in range(size):
                im = ax[cnl].imshow(inp[size * t + cnl])
                fig.colorbar(im, ax=ax[cnl])

            plt.savefig(os.path.join(outdir, f"target_{idx_inp}_{t}_" + ".png"))


def main():
    valid_loader = prepare_data_loaders()
    print(evaluate_on_val(valid_loader))


if __name__ == "__main__":
    main()
