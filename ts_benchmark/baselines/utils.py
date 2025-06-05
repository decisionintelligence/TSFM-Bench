# -*- coding: utf-8 -*-
import datetime
import random
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
import math

from ts_benchmark.baselines.time_series_library.utils.timefeatures import (
    time_features,
)
from ts_benchmark.utils.data_processing import split_before


class SlidingWindowDataLoader:
    """
    SlidingWindDataLoader class.

    This class encapsulates a sliding window data loader for generating time series training samples.
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        batch_size: int = 1,
        history_length: int = 10,
        prediction_length: int = 2,
        shuffle: bool = True,
    ):
        """
        Initialize SlidingWindDataLoader.

        :param dataset: Pandas DataFrame containing time series data.
        :param batch_size: Batch size.
        :param history_length: The length of historical data.
        :param prediction_length: The length of the predicted data.
        :param shuffle: Whether to shuffle the dataset.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.history_length = history_length
        self.prediction_length = prediction_length
        self.shuffle = shuffle
        self.current_index = 0

    def __len__(self) -> int:
        """
        Returns the length of the data loader.

        :return: The length of the data loader.
        """
        return len(self.dataset) - self.history_length - self.prediction_length + 1

    def __iter__(self) -> "SlidingWindowDataLoader":
        """
        Create an iterator and return.

        :return: Data loader iterator.
        """
        if self.shuffle:
            self._shuffle_dataset()
        self.current_index = 0
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate data for the next batch.

        :return: A tuple containing input data and target data.
        """
        if self.current_index >= len(self):
            raise StopIteration

        batch_inputs = []
        batch_targets = []
        for _ in range(self.batch_size):
            window_data = self.dataset.iloc[
                self.current_index : self.current_index
                + self.history_length
                + self.prediction_length,
                :,
            ]
            if len(window_data) < self.history_length + self.prediction_length:
                raise StopIteration  # Stop iteration when the dataset is less than one window size and prediction step size

            inputs = window_data.iloc[: self.history_length].values
            targets = window_data.iloc[
                self.history_length : self.history_length + self.prediction_length
            ].values

            batch_inputs.append(inputs)
            batch_targets.append(targets)
            self.current_index += 1

        # Convert NumPy array to PyTorch tensor
        batch_inputs = torch.tensor(batch_inputs, dtype=torch.float32)
        batch_targets = torch.tensor(batch_targets, dtype=torch.float32)

        return batch_inputs, batch_targets

    def _shuffle_dataset(self):
        """
        Shuffle the dataset.
        """
        self.dataset = self.dataset.sample(frac=1).reset_index(drop=True)


def train_val_split(train_data, ratio, seq_len):
    if ratio == 1:
        return train_data, None

    elif seq_len is not None:
        border = int((train_data.shape[0]) * ratio)

        train_data_value, valid_data_rest = split_before(train_data, border)
        train_data_rest, valid_data = split_before(train_data, border - seq_len)
        return train_data_value, valid_data
    else:
        border = int((train_data.shape[0]) * ratio)

        train_data_value, valid_data_rest = split_before(train_data, border)
        return train_data_value, valid_data_rest


def decompose_time(
    time: np.ndarray,
    freq: str,
) -> np.ndarray:
    """
    Split the given array of timestamps into components based on the frequency.

    :param time: Array of timestamps.
    :param freq: The frequency of the time stamp.
    :return: Array of timestamp components.
    """
    df_stamp = pd.DataFrame(pd.to_datetime(time), columns=["date"])
    freq_scores = {
        "m": 0,
        "w": 1,
        "b": 2,
        "d": 2,
        "h": 3,
        "t": 4,
        "s": 5,
    }
    max_score = max(freq_scores.values())
    df_stamp["month"] = df_stamp.date.dt.month
    if freq_scores.get(freq, max_score) >= 1:
        df_stamp["day"] = df_stamp.date.dt.day
    if freq_scores.get(freq, max_score) >= 2:
        df_stamp["weekday"] = df_stamp.date.dt.weekday
    if freq_scores.get(freq, max_score) >= 3:
        df_stamp["hour"] = df_stamp.date.dt.hour
    if freq_scores.get(freq, max_score) >= 4:
        df_stamp["minute"] = df_stamp.date.dt.minute
    if freq_scores.get(freq, max_score) >= 5:
        df_stamp["second"] = df_stamp.date.dt.second
    return df_stamp.drop(["date"], axis=1).values


def get_time_mark(
    time_stamp: np.ndarray,
    timeenc: int,
    freq: str,
) -> np.ndarray:
    """
    Extract temporal features from the time stamp.

    :param time_stamp: The time stamp ndarray.
    :param timeenc: The time encoding type.
    :param freq: The frequency of the time stamp.
    :return: The mark of the time stamp.
    """
    if timeenc == 0:
        origin_size = time_stamp.shape
        data_stamp = decompose_time(time_stamp.flatten(), freq)
        data_stamp = data_stamp.reshape(origin_size + (-1,))
    elif timeenc == 1:
        origin_size = time_stamp.shape
        data_stamp = time_features(pd.to_datetime(time_stamp.flatten()), freq=freq)
        data_stamp = data_stamp.transpose(1, 0)
        data_stamp = data_stamp.reshape(origin_size + (-1,))
    else:
        raise ValueError("Unknown time encoding {}".format(timeenc))
    return data_stamp.astype(np.float32)


def forecasting_data_provider(data, config, timeenc, batch_size, shuffle, drop_last, 
                              data_info = 'train', sampling_rate = 1, sampling_strategy = "uniform", sampling_basis = "sample"):
    dataset = DatasetForTransformer(
        dataset=data,
        history_len=config.seq_len,
        prediction_len=config.pred_len,
        label_len=config.label_len,
        timeenc=timeenc,
        freq=config.freq,
        # is_train=config.is_train,
        data_info=data_info,
        # lag_len=lag_len,
        # model_name=model_name,
        sampling_rate=sampling_rate,
        sampling_strategy=sampling_strategy,
        sampling_basis = sampling_basis
    )
    
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        drop_last=drop_last,
    )
    return dataset, data_loader


class DatasetForTransformer:
    def __init__(
        self,
        dataset: pd.DataFrame,
        history_len: int = 10,
        prediction_len: int = 2,
        label_len: int = 5,
        timeenc: int = 1,
        freq: str = "h",
        # is_train = True,
        data_info = "train",
        # lag_len = 0,
        sampling_rate = 1, 
        sampling_strategy = "uniform",
        sampling_basis = "sample",
        # model_name = 'moirai', # 最终须删除，修改其他模型，
    ):
        # init
        # self.model_name = model_name # 最终须删除，修改其他模型
        # self.is_train = is_train
        self.dataset = dataset
        self.history_length = history_len
        self.prediction_length = prediction_len
        self.label_length = label_len
        self.current_index = 0
        self.timeenc = timeenc
        self.freq = freq
        
        self.data_info = data_info
        # self.lag_len = lag_len
        self.sampling_rate = sampling_rate  
        self.sampling_strategy = sampling_strategy
        self.sampling_basis = sampling_basis
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[data_info]

        self.internal = 1
        if self.set_type == 0:
            if self.sampling_strategy == "uniform" or self.sampling_strategy == "random":
                self.internal = int(1 // self.sampling_rate)
        # else:
        self.__read_data__()

    def __len__(self) -> int:
        """
        Returns the length of the data loader.

        :return: The length of the data loader.
        """
        # return len(self.dataset) - self.history_length - self.prediction_length + 1
        if self.set_type == 0 and self.sampling_basis == 'sample':
            return int(self.n_timepoint * self.sampling_rate)   
        else:
            return self.n_timepoint

    def __read_data__(self):
        
        # x = self.dataset[-(self.sampling_rate*len(self.dataset)):].reset_index()
        if self.sampling_basis == 'sample': # 样本采样
            self.n_timepoint = len(self.dataset) - self.history_length - self.prediction_length + 1 # 样本数量
            if self.set_type == 0:
                # cho = math.ceil(self.sampling_rate * self.n_timepoint) + self.history_length + self.prediction_length - 1
                if self.sampling_strategy == "end":
                    self.df_stamp = self.dataset[-(math.ceil(self.sampling_rate * self.n_timepoint)+self.history_length + self.prediction_length - 1):].reset_index()
                    self.dataset = self.dataset[-(math.ceil(self.sampling_rate * self.n_timepoint)+self.history_length + self.prediction_length - 1):]
                elif self.sampling_strategy == "begin":
                    self.df_stamp = self.dataset[:math.ceil(self.sampling_rate * self.n_timepoint)+self.history_length + self.prediction_length - 1].reset_index()
                    self.dataset = self.dataset[:math.ceil(self.sampling_rate * self.n_timepoint)+self.history_length + self.prediction_length - 1]
                else:
                    self.df_stamp = self.dataset.reset_index()
            else:
                self.df_stamp = self.dataset.reset_index()
        else: # 数据点采样
            self.n_timepoint = math.ceil(self.sampling_rate * len(self.dataset)) - self.history_length - self.prediction_length + 1 # 已经few-shot后的样本数量
            if self.set_type == 0:
                if self.sampling_strategy == "end":
                    self.df_stamp = self.dataset[-math.ceil(self.sampling_rate * len(self.dataset)):].reset_index()
                    self.dataset = self.dataset[-math.ceil(self.sampling_rate * len(self.dataset)):]
                elif self.sampling_strategy == "begin":
                    self.df_stamp = self.dataset[:math.ceil(self.sampling_rate * len(self.dataset))].reset_index()
                    self.dataset = self.dataset[:math.ceil(self.sampling_rate * len(self.dataset))]
                else:
                    self.df_stamp = self.dataset.reset_index()
            else:
                self.df_stamp = self.dataset.reset_index()

        self.df_stamp = self.df_stamp[["date"]].values.transpose(1, 0)
        data_stamp = get_time_mark(self.df_stamp, self.timeenc, self.freq)[0]
        self.data_stamp = data_stamp
        self.time_stamp = self.df_stamp

        # self.n_var = self.dataset.shape[-1]
        
        

    def __getitem__(self, index):
        # s_begin = index
        # s_end = s_begin + self.history_length
        # r_begin = s_end - self.label_length
        # r_end = r_begin + self.label_length + self.prediction_length
        if self.set_type == 0:
            if self.sampling_strategy == "random": # 随机样本采样
                temp_internal = random.randint(1, self.internal)
                index = index * temp_internal
            elif self.sampling_strategy == "uniform": # 均匀样本采样 
                index = index * self.internal
            # else: end, begin 初始. 末端
                

        # c_begin = index // self.n_timepoint  # select variable
        # if self.model_name == 'moirai': # 最终须删除，修改其他模型
        #     s_begin = index - self.lag_len
        # else:
        #     s_begin = index
        # s_end = s_begin + self.history_length
        # if self.model_name == 'Moirai' and self.set_type == 2:
        #     # index = index + self.lag_len
        #     s_begin = index
        #     s_end = s_begin + self.history_length
        # else:
        s_begin = index
        s_end = s_begin + self.history_length
        r_begin = s_end - self.label_length
        r_end = r_begin + self.label_length + self.prediction_length

        seq_x = self.dataset[s_begin:s_end]
        seq_y = self.dataset[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        seq_x = torch.tensor(seq_x.values, dtype=torch.float32)
        seq_y = torch.tensor(seq_y.values, dtype=torch.float32)
        seq_x_mark = torch.tensor(seq_x_mark, dtype=torch.float32)
        seq_y_mark = torch.tensor(seq_y_mark, dtype=torch.float32)
        
        # seq_x_stamp = self.time_stamp[0][s_begin:s_end]
        # seq_y_stamp = self.time_stamp[0][s_end:r_end]
        # if isinstance(seq_x_stamp[0], pd.Timestamp):
        #     seq_x_stamp = np.array(seq_x_stamp, dtype='datetime64[ns]')
        #     seq_y_stamp = np.array(seq_y_stamp, dtype='datetime64[ns]')

        # seq_x_stamp = torch.tensor(seq_x_stamp.astype(float), dtype=torch.int64)
        # seq_y_stamp = torch.tensor(seq_y_stamp.astype(float), dtype=torch.int64)

        return seq_x, seq_y, seq_x_mark, seq_y_mark
        # return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_x_stamp, seq_y_stamp

class SegLoader(object):
    def __init__(self, data, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.data = data
        self.test_labels = data

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return (self.data.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "val":
            return (self.data.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "test":
            return (self.data.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.data.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.data[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.data[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.data[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.data[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


def anomaly_detection_data_provider(data, batch_size, win_size=100, step=100, mode='train'):
    dataset = SegLoader(data, win_size, 1, mode)

    shuffle = False
    if mode == "train" or mode == "val":
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0,
                             drop_last=False)
    return data_loader