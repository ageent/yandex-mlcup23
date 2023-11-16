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
from src.unet_reg import UNetModelReg


def prepare_data_loaders(train_batch_size=1, valid_batch_size=1, test_batch_size=1):
    valid_dataset = ProcessedRadarDataset(
        file_path="train.hdf5", preparation="multi_label", transform=init_aug()
    )

    valid_loader = data.DataLoader(
        valid_dataset,
        batch_size=valid_batch_size,
        num_workers=10,
        pin_memory=True,
        shuffle=False,
    )

    return valid_loader


def evaluate_on_val(model, valid_loader):
    outdir = "./plt_vis/" + str(datetime.now())
    os.makedirs(outdir, exist_ok=True)
    rmses = np.zeros((12,), dtype=float)
    for item in tqdm.tqdm(valid_loader):
        # inputs, target = next(iter(valid_loader))
        inputs, target = item
        target = np.array(target)
        output = model(inputs.to("cuda")).detach().cpu().numpy()
        output_for_class = []
        for i in range(target.shape[1]):
            output_for_class.append(np.argmax(output[0, 4 * i : 4 * i + 4], axis=0))
        output_for_class = np.array(output_for_class)
        output_for_class[target[0] == -1] = -1
        target = np.squeeze(target, axis=0)
        print(np.sum(target[target == 3]))
        print(np.sum(output_for_class[output_for_class == 3]))
        # for idx in range(target.shape[1]):
        #     plt.imsave(os.path.join(outdir, 'target_' + str(idx) + '.png'), target[0][idx])
        #     plt.imsave(os.path.join(outdir, 'predict_' + str(idx) + '.png'), output_for_class[idx])

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

    elif model_name == "unet_classify":
        model = UNetModelClassify()
    else:
        print("Unknown model name")

    model.load_state_dict(torch.load(checkpoint)["state_dict"])
    model.eval()
    model.to("cuda")
    print(evaluate_on_val(model, valid_loader))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    main(args.model, args.checkpoint)
