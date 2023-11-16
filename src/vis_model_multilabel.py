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
        file_path="val.hdf5",
        preparation="regression",
    )

    valid_loader = data.DataLoader(
        valid_dataset,
        batch_size=valid_batch_size,
        num_workers=10,
        pin_memory=True,
        shuffle=False,
    )

    return valid_loader


def inverse(data, min, max):
    mask = data == -1
    data = data * (max - min) + min
    data[data < min] = 0
    print(mask)
    data[mask] = -1
    return data


def evaluate_on_val(model, valid_loader):
    outdir = "./plt_vis/" + str(datetime.now())
    os.makedirs(outdir, exist_ok=True)
    # for item in tqdm.tqdm(valid_loader):
    iter_dt = iter(valid_loader)
    # inputs, target = next(iter_dt)
    for i in range(800):
        inputs, target = next(iter_dt)
    # inputs, target = item
    target = np.array(target)
    output = torch.sigmoid(model(inputs.to("cuda"))).detach().cpu().numpy()
    output[target == -1] = -1
    for i in range(12):
        output[:, 3 * i + 1][output[:, 3 * i] < 0.7] = 0
        output[:, 3 * i + 2][output[:, 3 * i + 1] < 0.7] = 0
    # output_for_class = []
    # for i in range(target.shape[1]):
    #     output_for_class.append(np.argmax(output[0,4*i:4*i+4], axis=0))
    # output_for_class = np.array(output_for_class)
    target = np.squeeze(target, axis=0)
    output = np.squeeze(output, axis=0)

    # output[target==-1] = -1
    print(np.sum(target))
    # print(np.sum(target[target==3]))
    # print(np.sum(output_for_class[output_for_class==3]))
    for idx in range(12):
        plt.imsave(
            os.path.join(outdir, "target_" + str(idx) + ".png"),
            target[3 * idx] + target[3 * idx + 1] + target[3 * idx + 2],
        )
        plt.imsave(
            os.path.join(outdir, "predict_" + str(idx) + ".png"),
            inverse(output[3 * idx], 0, 1)
            + inverse(output[3 * idx + 1], 1, 4)
            + inverse(output[3 * idx + 2], 4, 50),
        )

    # return np.mean(np.sqrt(rmses))


def main(model_name, checkpoint):
    valid_loader = prepare_data_loaders()
    if model_name == "persistant":
        # score on valid set: 197.64139689523992
        # score on test set: 283.66210850104176
        model = PersistantModel()
        print(evaluate_on_val(model, valid_loader))
        # process_test(model, test_loader)
    elif model_name == "convlstm":
        model = ConvLSTMModel()

    elif model_name == "unet_multi":
        model = UNetModelMulti()
    else:
        print("Unknown model name")
    model_weights = torch.load(checkpoint)["state_dict"]
    model.load_state_dict(model_weights)
    model.eval()
    model.to("cuda")
    print(evaluate_on_val(model, valid_loader))


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
    main(args.model, args.checkpoint)
