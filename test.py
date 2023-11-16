import os
from datetime import datetime
from pathlib import Path

import h5py
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
import tqdm

from src.datasets import (FullFeatRegSegDataset, ProcessedRadarDataset,
                          RadarDataset)
from src.models import ConvLSTMModel, PersistantModel
from src.preparation import boxcox_func
from src.unet_classify import UNetModelClassify
from src.unet_garynych import UNetModelGarynych
from src.unet_reg import UNetModelReg
from src.unet_reg_tune import UNetModelReg
from src.unet_two_head import UNetModelTwoHead
from src.vis_garynych import stack_res


def prepare_data_loaders(
    path_to_data, train_batch_size=1, valid_batch_size=1, test_batch_size=1
):
    valid_dataset = FullFeatRegSegDataset(
        path_to_data, out_seq_len=0, target_source=True, with_time=True
    )

    valid_loader = data.DataLoader(
        valid_dataset,
        batch_size=valid_batch_size,
        num_workers=10,
        pin_memory=True,
        shuffle=False,
    )

    return valid_loader


def process_test(model, test_loader, output_file="out/garynych_129_0_65_trs.hdf5"):
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    for index, item in tqdm.tqdm(enumerate(test_loader)):
        ((velocity, intensity, reflectivity, events), last_input_timestamp), _ = item
        output = model(
            (
                velocity.to("cuda"),
                intensity.to("cuda"),
                reflectivity.to("cuda"),
                events.to("cuda"),
            )
        )
        out_seg, out_reg = output
        out_seg = torch.sigmoid(out_seg).detach().cpu()
        out_reg = torch.sigmoid(out_reg).detach().cpu()
        out_reg = torch.squeeze(out_reg)
        out_seg = torch.squeeze(out_seg)

        out_seg = out_seg > 0.7
        output = np.array(stack_res(out_seg, out_reg, to_abs=True))

        output = np.expand_dims(output, 0)
        output = np.expand_dims(output, axis=2)

        with h5py.File(output_file, mode="a") as f_out:
            for index in range(output.shape[1]):
                timestamp_out = str(int(last_input_timestamp[-1]) + 600 * (index + 1))
                f_out.create_group(timestamp_out)
                f_out[timestamp_out].create_dataset(
                    "intensity", data=output[0, index, 0]
                )


def main():
    test_loader = prepare_data_loaders("data/2022-test-public.hdf5")
    checkpoint = "chkpt/unet_garynych_trainval_epoch=5.ckpt"
    model = UNetModelGarynych()
    model.load_state_dict(torch.load(checkpoint)["state_dict"])
    model.eval()

    import ttach.ttach as tta
    import ttach.ttach.user_transforms as tta_tr

    transforms = tta.Compose(
        [
            tta_tr.Upscale(),
            tta_tr.Downscale(),
            tta_tr.Rotation(),
        ]
    )
    model = tta.SegmentationTTAWrapper(model, transforms, merge_mode='sparse_mean')

    model.to("cuda")
    process_test(model, test_loader)


if __name__ == "__main__":
    main()
