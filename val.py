import argparse

import h5py
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
import tqdm

from src.datasets import (IntensityRegSegDataset, ProcessedRadarDataset,
                          RadarDataset)
from src.models import ConvLSTMModel, PersistantModel
from src.preparation import boxcox_func
from src.unet_classify import UNetModelClassify
from src.unet_reg import UNetModelReg
from src.unet_reg_tune import UNetModelReg
from src.unet_two_head import UNetModelTwoHead
from src.vis_two_head import stack_res


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


def evaluate_on_val(model, valid_loader):
    rmses = np.zeros((12,), dtype=float)
    for item in tqdm.tqdm(valid_loader):
        inputs, target = item
        target = np.array(target)
        output = model(inputs.to("cuda"))
        out_seg, out_reg = output
        out_seg = torch.sigmoid(out_seg).detach().cpu()
        out_reg = torch.sigmoid(out_reg).detach().cpu()
        out_reg = torch.squeeze(out_reg)
        out_seg = torch.squeeze(out_seg)

        trsh = 0.5
        out_seg = out_seg > trsh
        out_seg[2::3] = 0
        output = stack_res(out_seg, out_reg, to_abs=True)

        output = np.expand_dims(np.array(output), 0)
        target = np.array(target)

        output = np.expand_dims(output, axis=2)
        target = np.expand_dims(target, axis=2)

        rmses += np.sum(
            (np.square(target - output)) * (target != -1), axis=(0, 2, 3, 4)
        )
    rmses /= len(valid_loader)
    return np.mean(np.sqrt(rmses))


def process_test(model, test_loader, output_file="../output.hdf5"):
    model.eval()
    model.to("cuda")
    for index, item in tqdm.tqdm(enumerate(test_loader)):
        (inputs, last_input_timestamp), _ = item
        output = model(inputs.to("cuda")).cpu()
        with h5py.File(output_file, mode="a") as f_out:
            for index in range(output.shape[1]):
                timestamp_out = str(int(last_input_timestamp[-1]) + 600 * (index + 1))
                f_out.create_group(timestamp_out)
                f_out[timestamp_out].create_dataset(
                    "intensity", data=output[0, index, 0].detach().numpy()
                )


def main(model_name, checkpoint):
    valid_loader = prepare_data_loaders()
    if model_name == "persistant":
        # score on valid set: 197.64139689523992
        # score on our valid set: 220.3429690748951
        # score on test set: 283.66210850104176
        model = PersistantModel()
        print(evaluate_on_val(model, valid_loader))
        # process_test(model, test_loader)
    elif model_name == "convlstm":
        model = ConvLSTMModel()

    elif model_name == "unet_classify":
        model = UNetModelClassify(num_iters=len(train_loader))
        # score on our valid set: 275.44051926751007
    elif model_name == "unet_two_head":
        model = UNetModelTwoHead()
    else:
        print("Unknown model name")

    model.load_state_dict(torch.load(checkpoint)["state_dict"])
    model.eval()
    model.to("cuda")
    print(evaluate_on_val(model, valid_loader))
    # process_test(model, test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument(
        "--checkpoint",
        default="checkpoint/2023-11-10 00:04:34.487578/unet_two_head_epoch=54_val_loss=0.132021.ckpt",
    )
    args = parser.parse_args()
    main(args.model, args.checkpoint)
