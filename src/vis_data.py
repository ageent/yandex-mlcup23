import os
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torch.utils.data as data

from src.augmentation import init_aug
from src.datasets import ProcessedRadarDataset, RadarDataset

# torch.multiprocessing.set_start_method('spawn')


def prepare_data():
    dataset_aurgment = ProcessedRadarDataset(
        file_path="val.hdf5", preparation="regression", transform=init_aug()
    )

    dataset_src = ProcessedRadarDataset(file_path="val.hdf5", preparation="tensor")
    train_loader_aug = data.DataLoader(
        dataset_aurgment, batch_size=1, num_workers=10, pin_memory=True, shuffle=False
    )
    train_loader_src = data.DataLoader(
        dataset_src, batch_size=1, num_workers=10, pin_memory=True, shuffle=False
    )
    return train_loader_aug, train_loader_src


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    outdir = "./plt_vis/" + str(datetime.now())
    os.makedirs(outdir, exist_ok=True)
    train_loader_aug, train_loader_src = prepare_data()
    input, target = next(iter(train_loader_aug))
    input, target = input[0], target[0]
    for idx in range(4):
        plt.imsave(
            os.path.join(outdir, "input_aug_" + str(idx) + ".png"),
            input[3 * idx] + input[3 * idx + 1] + input[3 * idx + 2],
        )

    for idx in range(12):
        plt.imsave(
            os.path.join(outdir, "target_aug_" + str(idx) + ".png"),
            target[3 * idx] + target[3 * idx + 1] + target[3 * idx + 2],
        )

    # input, target = next(iter(train_loader_src))
    # input, target = input[0], target[0]
    # for idx in range(input.shape[0]):
    #     plt.imsave(os.path.join(outdir, 'input_src_' + str(idx) + '.png'), input[idx])
    # for idx in range(target.shape[0]):
    #     plt.imsave(os.path.join(outdir, 'target_src_' + str(idx) + '.png'), target[idx])

# input, target = dataset_aurgment[0]
