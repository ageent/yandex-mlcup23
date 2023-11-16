import argparse
import os

import h5py
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data
import tqdm
from torchmetrics.classification import BinaryJaccardIndex

from src.datasets import (FullFeatRegSegDataset, IntensityRegSegDataset,
                          ProcessedRadarDataset)
from src.unet_garynych import UNetModelGarynych


def prepare_data_loaders(train_batch_size=1, valid_batch_size=50, test_batch_size=1):
    valid_dataset = FullFeatRegSegDataset(file_path="val_full.hdf5")

    valid_loader = data.DataLoader(
        valid_dataset,
        batch_size=valid_batch_size,
        num_workers=20,
        pin_memory=True,
        shuffle=False,
    )

    return valid_loader


def search_trsh(model, valid_loader, trsh, chnl=1):
    values = []
    metrics = BinaryJaccardIndex(threshold=trsh).to("cuda")
    for item in tqdm.tqdm(valid_loader):
        (velocity, intensity, reflectivity, events), (y_seg, y_reg) = item
        out_seg, out_reg = model(
            (
                velocity.to("cuda"),
                intensity.to("cuda"),
                reflectivity.to("cuda"),
                events.to("cuda"),
            )
        )
        out_seg = torch.sigmoid(out_seg.detach())
        # out_seg[:,1::3][out_seg[:,0::3] < 0.75] = 0
        # out_seg[:,2::3][out_seg[:,1::3] < 0.70] = 0
        y_seg[:, 0::3][y_seg[:, 0::3] == -1] = 0
        res = metrics(out_seg[:, 0::3], y_seg[:, 0::3].to("cuda")).cpu()
        res = torch.nan_to_num(res, nan=0)
        values.append(float(res))
    values = [i for i in values if i != 0]
    return sum(values) / len(values)


def main():
    valid_loader = prepare_data_loaders()
    checkpoint = "checkpoint/2023-11-12 00:35:52.929493/unet_garynych_epoch=31_val_loss=0.133835.ckpt"
    model = UNetModelGarynych()
    model.load_state_dict(torch.load(checkpoint)["state_dict"])
    model.eval()
    model.to("cuda")
    trsh = 0.7
    # for trsh in np.arange(0.s4, 0.9, 0.1):
    error = search_trsh(model, valid_loader, trsh)
    print(trsh, error)


if __name__ == "__main__":
    main()
