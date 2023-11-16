import argparse

import h5py
import lightning as L
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
from src.unet_garynych import UNetModelGarynych
from src.unet_multi_label import UNetModelMulti
from src.unet_multi_label_iter import UNetModelMultiIter
from src.unet_reg import UNetModelReg
from src.unet_reg_tune import UNetModelReg
from src.unet_two_head import UNetModelTwoHead


def prepare_data_loaders(
    train_path,
    val_path,
    train_batch_size=60,
    valid_batch_size=1,
    test_batch_size=1,
    preparation="reg_segment",
):
    train_dataset = FullFeatRegSegDataset(file_path=train_path)
    valid_dataset = FullFeatRegSegDataset(file_path=val_path)

    transforms = init_aug()
    train_loader = FullFeatRegSegDataLoader(
        transforms,
        train_dataset,
        batch_size=train_batch_size,
        num_workers=30,
        shuffle=True,
    )
    valid_loader = data.DataLoader(
        valid_dataset, batch_size=valid_batch_size, num_workers=0, shuffle=False
    )
    return train_loader, valid_loader


def main(model_name, tensorboard_path, preparation):
    torch.manual_seed(42)
    train_loader, valid_loader = prepare_data_loaders(
        "data/train.hdf5",
        "data/val.hdf5",
        preparation=preparation,
    )
    trainer, num_epoch = init_trainer(model_name, tensorboard_path)

    if model_name == "persistant":
        # score on valid set: 197.64139689523992
        # score on test set: 283.66210850104176
        model = PersistantModel()
    elif model_name == "convlstm":
        model = ConvLSTMModel()
    elif model_name == "unet_reg":
        model = UNetModelReg(num_epoch=num_epoch, num_iters=len(train_loader))
    elif model_name == "unet_classify":
        model = UNetModelClassify(num_epoch=num_epoch, num_iters=len(train_loader))
    elif model_name == "unet_multi":
        model = UNetModelMulti(num_epoch=num_epoch, num_iters=len(train_loader))
    elif model_name == "unet_multi_iter":
        model = UNetModelMultiIter(num_epoch=num_epoch, num_iters=len(train_loader))
    elif model_name == "unet_two_head":
        model = UNetModelTwoHead(num_epoch=num_epoch, num_iters=len(train_loader))
    elif model_name == "unet_reg_tune":
        model = UNetModelReg(num_epoch=num_epoch, num_iters=len(train_loader))
        checkpoint = "chkpt/unet_multi_iter_epoch=23_val_loss=0.637777.ckpt"
        model_weights = torch.load(checkpoint)["state_dict"]
        model.load_state_dict(model_weights)
        model.freeze_weights()
    elif model_name == "unet_garynych":
        model = UNetModelGarynych(num_epoch=num_epoch, num_iters=len(train_loader))
        # checkpoint = 'chkpt/unet_garynych_epoch=38_val_loss=0.134067.ckpt'
        # model_weights = torch.load(checkpoint)['state_dict']
        # model.load_state_dict(model_weights)
    else:
        print("Unknown model name")

    trainer.fit(model, train_loader, valid_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--tensorboard_path", default="./tensorboard")
    parser.add_argument("--preparation", default="regression")
    args = parser.parse_args()
    main(args.model, args.tensorboard_path, args.preparation)
