import h5py
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import torchvision
from kornia.morphology import dilation

from src.preparation import boxcox_func

torchvision.disable_beta_transforms_warning()


class TransformedDataLoader:
    def __init__(self, transforms, *args, **kwargs):
        self.data_loader = data.DataLoader(*args, **kwargs)
        self.transforms = transforms

    def __iter__(self):
        for inp, out in iter(self.data_loader):
            x = torch.cat((inp, out), 1)
            if self.transforms:
                x = self.transforms(x)
            yield x[:, : inp.shape[1]], x[:, inp.shape[1] :]

    def __len__(self):
        return len(self.data_loader)


class RegSegTransformedDataLoader(TransformedDataLoader):
    def __iter__(self):
        for inp, (seg, reg) in iter(self.data_loader):
            x = torch.cat((inp, seg, reg), 1)
            if self.transforms:
                x = self.transforms(x)

            inp_c = inp.shape[1]
            inp_seg_c = seg.shape[1] + inp_c
            yield x[:, :inp_c], (x[:, inp_c:inp_seg_c], x[:, inp_seg_c:])


class FullFeatRegSegDataLoader(TransformedDataLoader):
    def __iter__(self):
        isnt_precalculated = True
        for inps, outs in iter(self.data_loader):
            x = torch.cat((*inps, *outs), 1).to("cuda")
            if self.transforms:
                x = self.transforms(x)

            if isnt_precalculated:
                cnls_inp = [inps[i].shape[1] for i in range(len(inps))]
                cnls_inp_sum = sum(cnls_inp)
                cnls = cnls_inp + [outs[i].shape[1] for i in range(len(outs))]
                cnls[1:] = [cnls[i] + sum(cnls[:i]) for i in range(1, len(cnls))]

            nanmask = x.isnan()
            x[:, :cnls_inp_sum][nanmask[:, :cnls_inp_sum]] = 0
            x[:, cnls_inp_sum:][nanmask[:, cnls_inp_sum:]] = -1

            yield (
                x[:, : cnls[0]],
                x[:, cnls[0] : cnls[1]],
                x[:, cnls[1] : cnls[2]],
                x[:, cnls[2] : cnls[3]],
            ), (x[:, cnls[3] : cnls[4]], x[:, cnls[4] : cnls[5]])


class RadarDataset(data.Dataset):
    def __init__(
        self,
        list_of_files,
        in_seq_len=4,
        out_seq_len=12,
        mode="overlap",
        with_time=False,
        preparation="tensor",  # data preparation method: None, boxcox, tensor, classify
        transform=None,
        target_source=False,
    ):
        self.in_seq_len = in_seq_len
        self.out_seq_len = out_seq_len
        self.seq_len = in_seq_len + out_seq_len
        self.with_time = with_time
        self.prepare_method = preparation
        self.target_source = target_source
        self.transform = transform
        self.__prepare_timestamps_mapping(list_of_files)
        self.__prepare_sequences(mode)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        data = []
        for timestamp in self.sequences[index]:
            with h5py.File(self.timestamp_to_file[timestamp]) as d:
                data.append(np.array(d[timestamp]["intensity"]))

        data = np.array(data)
        data[data == -1e6] = 0
        data[data == -2e6] = -1
        data = torch.Tensor(data)

        inputs = data[: self.in_seq_len]
        targets = data[self.in_seq_len :]
        inputs = self.prepare(inputs)
        if self.target_source == False:
            targets = self.prepare(targets, type="target")

        if self.with_time:
            return inputs, self.sequences[index][-1]
        else:
            return inputs.detach(), targets.detach()

    def __prepare_timestamps_mapping(self, list_of_files):
        self.timestamp_to_file = {}
        for filename in list_of_files:
            with h5py.File(filename, "r") as d:
                self.timestamp_to_file = {
                    **self.timestamp_to_file,
                    **dict(map(lambda x: (x, filename), d.keys())),
                }

    def __prepare_sequences(self, mode):
        timestamps = np.unique(sorted(self.timestamp_to_file.keys()))
        if mode == "sequentially":
            self.sequences = [
                timestamps[index * self.seq_len : (index + 1) * self.seq_len]
                for index in range(len(timestamps) // self.seq_len)
            ]
        elif mode == "overlap":
            self.sequences = [
                timestamps[index : index + self.seq_len]
                for index in range(len(timestamps) - self.seq_len + 1)
            ]
        else:
            raise Exception(f"Unknown mode {mode}")
        self.sequences = list(
            filter(
                lambda x: int(x[-1]) - int(x[0]) == (self.seq_len - 1) * 600,
                self.sequences,
            )
        )

    def prepare(self, data, type="inputs"):
        if self.prepare_method == "boxcox":
            return boxcox_func(data)
        elif self.prepare_method == "tensor":
            return torch.Tensor(data)
        elif self.prepare_method == "classify":
            if type == "inputs":
                new_data = torch.zeros(
                    (data.shape[0] * 4, data.shape[1], data.shape[2])
                ).to(data.device)
                for idx in range(data.shape[0]):
                    new_data[4 * idx][data[idx] == 0] = 1
                    new_data[4 * idx + 1][(data[idx] > 0) & (data[idx] < 1)] = 1
                    new_data[4 * idx + 2][(data[idx] >= 1) & (data[idx] < 4)] = 1
                    new_data[4 * idx + 3][data[idx] >= 4] = 1
                return new_data
            else:
                new_data = torch.full(
                    (data.shape[0], data.shape[1], data.shape[2]), fill_value=-1
                ).to(data.device)
                for idx in range(data.shape[0]):
                    new_data[idx][data[idx] == 0] = 0
                    new_data[idx][(data[idx] > 0) & (data[idx] < 1)] = 1
                    new_data[idx][(data[idx] >= 1) & (data[idx] < 4)] = 2
                    new_data[idx][data[idx] >= 4] = 3
                return new_data

        elif self.prepare_method == "regression":
            if type == "inputs":
                new_data = torch.zeros(
                    (data.shape[0] * 3, data.shape[1], data.shape[2])
                ).to(data.device)
                for idx in range(data.shape[0]):
                    new_data[3 * idx] = self.get_interval(data[idx].clone(), 0, 1)
                    new_data[3 * idx + 1] = self.get_interval(data[idx].clone(), 1, 4)
                    new_data[3 * idx + 2] = self.get_interval(data[idx].clone(), 4, 50)
                return new_data
            else:
                new_data = torch.zeros(
                    (data.shape[0] * 3, data.shape[1], data.shape[2])
                ).to(data.device)
                for idx in range(data.shape[0]):
                    new_data[3 * idx] = self.get_interval(data[idx].clone(), 0, 1)
                    new_data[3 * idx + 1] = self.get_interval(data[idx].clone(), 1, 4)
                    new_data[3 * idx + 2] = self.get_interval(data[idx].clone(), 4, 50)
                    for i in range(3):
                        new_data[3 * idx + i][data[idx] == -1] = -1

                return new_data

        elif self.prepare_method == "reg_segment":
            thresholds = torch.tensor([0, 1, 4, 50], dtype=float)

            c, h, w = data.shape
            min_trs = thresholds[:-1]
            max_trs = thresholds[1:]
            scale_c = min_trs.shape[0]
            divider = max_trs - min_trs

            divider[-1] /= 3

            divider = divider.view(-1, 1, 1, 1)
            min_trs = min_trs.view(-1, 1, 1, 1)
            max_trs = max_trs.view(-1, 1, 1, 1)

            res = (
                data.repeat(scale_c, 1, 1, 1)
                .clip_(min=min_trs, max=max_trs)
                .sub_(min_trs)
                .div_(divider)
            )

            return res.transpose_(0, 1).reshape(-1, h, w)
        else:
            return data


class ProcessedRadarDataset(RadarDataset):
    def __init__(
        self,
        file_path,
        in_seq_len=4,
        out_seq_len=12,
        with_time=False,
        preparation="tensor",  # data preparation method: None, boxcox, tensor, classify
        target_source=False,
        transform=None,
    ):
        self.file_path = file_path
        self.in_seq_len = in_seq_len
        self.out_seq_len = out_seq_len
        self.seq_len = in_seq_len + out_seq_len
        self.with_time = with_time
        self.prepare_method = preparation
        self.target_source = target_source
        self.transform = transform
        self.__prepare_timestamps(file_path)
        self.__prepare_sequences()

    def __getitem__(self, index):
        data = []
        for timestamp in self.sequences[index]:
            with h5py.File(self.file_path, "r") as d:
                data.append(np.array(d[str(timestamp)]["intensity"]))

        data = np.asarray(data)
        data[data == -1e6] = 0
        data[data == -2e6] = -1
        data = torch.Tensor(data)

        if self.transform:
            data = self.transform(data)

        inputs = data[: self.in_seq_len]
        targets = data[self.in_seq_len :]

        inputs = self.prepare(inputs)
        if self.target_source == False:
            targets = self.prepare(targets, type="target")

        if self.with_time:
            return (inputs, self.sequences[index][-1]), targets
        else:
            return inputs.detach(), targets.detach()

    def __prepare_timestamps(self, file_path):
        timestamps = []
        first_input = []
        with h5py.File(file_path, "r") as d:
            for timestamp in d.keys():
                timestamps.append(int(timestamp))
                if self.out_seq_len:
                    first_input.append(d[timestamp].attrs["first_input"])
                else:
                    first_input.append(True)

        self.first_input_by_timestamp = pd.Series(
            first_input,
            index=timestamps,
        ).sort_index()

    def __prepare_sequences(self):
        # mode: overlap
        timestamps = self.first_input_by_timestamp.index.values

        seq = []
        for i in range(timestamps.size - self.seq_len + 1):
            if self.first_input_by_timestamp.iloc[i]:
                seq.append(timestamps[i : i + self.seq_len])

        seq = np.asarray(seq)
        seq_time_period = seq[:, -1] - seq[:, 0]
        correct_seq_time_period = (self.seq_len - 1) * 600

        self.sequences = seq[seq_time_period == correct_seq_time_period]


class IntensityRegSegDataset(ProcessedRadarDataset):
    def __init__(
        self,
        file_path,
        in_seq_len=4,
        out_seq_len=12,
        with_time=False,
        target_source=False,
        device="cpu",
    ):
        super().__init__(
            file_path,
            in_seq_len=in_seq_len,
            out_seq_len=out_seq_len,
            with_time=with_time,
            preparation="reg_segment",
            target_source=target_source,
            transform=None,
        )
        self.device = device

    def __getitem__(self, index):
        data = []
        for timestamp in self.sequences[index]:
            with h5py.File(self.file_path, "r") as d:
                data.append(d[str(timestamp)]["intensity"][:])

        data = (
            torch.from_numpy(np.asarray(data).astype(np.float32))
            .requires_grad_(False)
            .to(self.device)
        )
        data[data == -1e6] = 0
        data[data == -2e6] = -1

        inputs = data[: self.in_seq_len]
        targets = data[self.in_seq_len :]

        if self.target_source == False:
            inputs, targets = self.prepare(inputs, targets)
        else:
            inputs = self.prepare(inputs)

        if self.with_time:
            return (inputs, self.sequences[index][-1]), targets
        else:
            return (
                (torch.cat((inputs, inputs[-8:])), inputs, inputs[-4:], inputs[-4:]),
                targets,
            )

    def prepare(self, inp, target=None):
        trs = torch.tensor([0.0, 0.5, 1.5, 5.0], dtype=float).to(self.device)

        res_inp = self._prepare_intensity(inp, trs)

        if target is None:
            return res_inp

        t = self._prepare_intensity(target, trs, inp=False)
        reg = self._prepare_terget_reg(t)
        seg = self._prepare_iter_enc(target, t)

        return res_inp, (seg, reg)

    def _prepare_intensity(self, x, thresholds, inp=True):
        c, h, w = x.shape
        min_trs = thresholds[:-1]
        max_trs = thresholds[1:]
        scale_c = min_trs.shape[0]
        divider = max_trs - min_trs

        if inp:
            divider[-1] /= 3

        divider = divider.view(-1, 1, 1, 1)
        min_trs = min_trs.view(-1, 1, 1, 1)
        max_trs = max_trs.view(-1, 1, 1, 1)

        res = (
            x.repeat(scale_c, 1, 1, 1)
            .clip_(min=min_trs, max=max_trs)
            .sub_(min_trs)
            .div_(divider)
        )

        if inp:
            return res.transpose_(0, 1).reshape(-1, h, w)
        return res

    def _prepare_terget_reg(self, x):
        x = x.transpose(0, 1).reshape(-1, x.shape[-2], x.shape[-1])
        kernel = torch.ones(3, 3).to(self.device)
        mask = dilation(x.unsqueeze(0), kernel).bool().squeeze()
        x[~mask] = -1
        return x

    def _prepare_iter_enc(self, src_data, x):
        # src_data.shape == c h w
        # x.shape == r c h w

        x = x.bool()

        mask0 = (src_data == -1).unsqueeze(0)
        mask12 = ~x[:-1]
        mask = torch.cat([mask0, mask12])

        x = x.float()
        x[mask] = -1
        return x.transpose_(0, 1).reshape(-1, x.shape[-2], x.shape[-1])


class FullFeatRegSegDataset(IntensityRegSegDataset):
    def __getitem__(self, index):
        with h5py.File(self.file_path, "r") as d:
            k = next(iter(d.keys()))
            feats = list(d[k].keys())
            data = {f: [] for f in d[k].keys()}

        with h5py.File(self.file_path, "r") as d:
            for feat in feats:
                size = self.in_seq_len
                if feat == "intensity":
                    size = None

                for timestamp in self.sequences[index][:size]:
                    data[feat].append(d[str(timestamp)][feat][:])

        # print(len(data[feats[0]]), len(data[feats[1]]), len(data[feats[2]]), len(data[feats[3]]))

        for feat in feats:
            d = data[feat]
            d = (
                torch.from_numpy(np.asarray(d))
                .float()
                .requires_grad_(False)
                .to(self.device)
            )

            empty_v = -1 if feat == "intensity" else 0
            d[d == -2e6] = empty_v
            d[d == -1e6] = 0

            data[feat] = d

        inputs, targets = self.prepare(data)

        if self.with_time:
            return (inputs, self.sequences[index][-1]), targets
        else:
            return inputs, targets

    def prepare(self, data):
        velocity = self._prepare_velocity_feat(data["radial_velocity"])
        reflectivity = self._prepare_reflectivity_feat(data["reflectivity"])
        events = self._prepare_events_feat(data["events"])

        # intensity
        intensity = data["intensity"][: self.in_seq_len]
        targets = data["intensity"][self.in_seq_len :]
        if self.target_source == False:
            intensity, targets = self._prepare_intensity_feat(intensity, targets)
        else:
            intensity = self._prepare_intensity_feat(intensity)

        return (velocity, intensity, reflectivity, events), targets

    def _prepare_velocity_feat(self, inp):
        _, _, h, w = inp.shape

        inds = torch.arange(0, 10, 2).to(self.device)

        inp = torch.stack([inp[:, inds], inp[:, inds + 1]], dim=1)
        dmax = inp.max(dim=1).values
        dmin = inp.min(dim=1).values

        res = torch.where(-dmin > dmax, dmin, dmax)

        return self._clip_zeronorm(res, 10.5).reshape(-1, h, w)

    def _prepare_reflectivity_feat(self, inp):
        return self._clip_zeronorm(inp.sum(dim=1), 71.5)

    def _prepare_events_feat(self, inp):
        return self._clip_zeronorm(inp, 6)

    def _prepare_intensity_feat(self, inp, target=None):
        return super().prepare(inp, target)

    def _clip_zeronorm(self, x, std):
        std3 = std * 3
        return x.clip_(min=-std3, max=std3).div_(std)
