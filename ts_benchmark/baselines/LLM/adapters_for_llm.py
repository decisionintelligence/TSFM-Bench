import os
from typing import Type, Dict, Optional, Tuple

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch import optim

from ts_benchmark.baselines.time_series_library.utils.tools import (
    EarlyStopping,
    adjust_learning_rate,
)
from ts_benchmark.baselines.utils import (
    forecasting_data_provider,
    train_val_split,
    get_time_mark,
)
from ts_benchmark.models.model_base import ModelBase, BatchMaker
from ts_benchmark.utils.data_processing import split_before

DEFAULT_LLM_BASED_HYPER_PARAMS = {
    "num_samples": 100,
    "quantiles_num": 20,
    "ckpt_path":"",
    "dataset":"etth1",
    "patience": 3,
    "num_epochs": 10,
    "lradj": "type1",
    "freq": "H",
    "batch_size": 64,
    'label_len':96,
    "num_workers": 0,
    "freq": "h",
    "sampling_rate": 0.05,
    "sampling_strategy": "uniform",
    "sampling_basis": "sample",
    "is_train": 0,
    "get_train": 0,
    "lr": 0.0001,

    "get_pt": 0,
    "is_gpt": 1, 
    "patch_size": 16,
    "kernel_size": 25,
    "pretrain": 1,
    "seq_len": 96,
    "horizon": 96,
    "stride": 8,
    "gpt_layers": 3,
    "d_model": 768,
    "freeze":1,

    "use_multi_gpu": 0, # gpt4ts
    "local_rank": 0,
    "mix_embeds": 1,
    "mlp_hidden_layers": 0,
    "mlp_hidden_dim": 256,
    "mlp_activation": 'tanh',
    "dropout": 0.1,
    "token_len": 96,
    "use_p": 0,

    "ln": 0, # s2lpllm
    "task_name": "long_term_forecast",
    "patch_size": 16,
    "stride": 8,
    "pretrained": 1,
    "pool_size": 1000,
    "prompt_length": 8,
    "trend_length": 96,
    "seasonal_length": 24,
    "prompt_init": 'text_prototype',

    "d_ff": 32, #timellm
    "llm_dim": 768,
    "patch_len": 16,
    "llm_model": "GPT2",
    "prompt_domain": 1,
    "n_heads": 8,
    "enc_in": 7,
    "llm_layers": 12,
    "content": "",

    "max_token_num": 50, # unitime
    "mask_rate": 0.5,
    "max_backcast_len": 96,
    "max_forecast_len": 720,
    "ts_embed_dropout": 0.3,
    "lm_ft_type": "fpt",
    "instruct_path": "dataset/instruct.json",
    "lm_layer_num": 6,
    "dec_trans_layer_num": 2, 
    "ts_embed_dropout": 0.3,
    "dec_head_dropout": 0.1,

    "top_k" : 5, #LLMMixer
    "num_kernels": 6,
    # "d_model": 16,
    # "n_heads": 4,
    "e_layers": 2,
    "d_layers": 1,
    "moving_avg": 25,
    "factor": 1,
    "distil": 1,
    "channel_independence": 1,
    "decomp_method": "moving_avg",
    "down_sampling_layers": 3,
    "down_sampling_window": 2,
    "down_sampling_method": "avg",
    "use_future_temporal_feature": 0,
    "llm_path": "ts_benchmark/baselines/LLM/checkpoints/roberta-base",
    "tokenizer_path": "ts_benchmark/baselines/LLM/checkpoints/roberta-base",
    "embed": "timeF",
    "description": "",
    "r": 8,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "word_embedding_path": "ts_benchmark/baselines/LLM/checkpoints/wte_pca_500.pt",
}


class LLMConfig:
    def __init__(self, **kwargs):
        for key, value in DEFAULT_LLM_BASED_HYPER_PARAMS.items():
            setattr(self, key, value)

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def pred_len(self):
        return self.horizon


class LLMAdapter(ModelBase):
    def __init__(self, model_name, model_class, **kwargs):
        super(LLMAdapter, self).__init__()
        self.config = LLMConfig(**kwargs)
        self._model_name = model_name
        self.model_class = model_class
        self.scaler = StandardScaler()
        self.seq_len = self.config.seq_len


    @staticmethod
    def required_hyper_params() -> dict:
        """
        Return the hyperparameters required by model.

        :return: An empty dictionary indicating that model does not require additional hyperparameters.
        """
        return {}

    @property
    def model_name(self):
        """
        Returns the name of the model.
        """

        return self._model_name

    def multi_forecasting_hyper_param_tune(self, train_data: pd.DataFrame):
        freq = pd.infer_freq(train_data.index)
        if freq == None:
            self.config.freq
            # raise ValueError("Irregular time intervals")
        elif freq[0].lower() not in ["m", "w", "b", "d", "h", "t", "s"]:
            self.config.freq = "s"
        else:
            self.config.freq = freq[0].lower()

        column_num = train_data.shape[1]
        self.config.enc_in = column_num
        self.config.dec_in = column_num
        self.config.c_out = column_num

        setattr(self.config, "label_len", self.config.seq_len // 2)

    def padding_data_for_forecast(self, test):
        time_column_data = test.index
        data_colums = test.columns
        start = time_column_data[-1]

        date = pd.date_range(
            start=start, periods=self.config.horizon + 1, freq=self.config.freq.upper()
        )
        df = pd.DataFrame(columns=data_colums)

        df.iloc[: self.config.horizon + 1, :] = 0

        df["date"] = date
        df = df.set_index("date")
        new_df = df.iloc[1:]
        test = pd.concat([test, new_df])
        return test

    def _padding_time_stamp_mark(
        self, time_stamps_list: np.ndarray, padding_len: int
    ) -> np.ndarray:
        """
        Padding time stamp mark for prediction.

        :param time_stamps_list: A batch of time stamps.
        :param padding_len: The len of time stamp need to be padded.
        :return: The padded time stamp mark.
        """
        padding_time_stamp = []
        for time_stamps in time_stamps_list:
            start = time_stamps[-1]
            expand_time_stamp = pd.date_range(
                start=start,
                periods=padding_len + 1,
                freq=self.config.freq.upper(),
            )
            padding_time_stamp.append(expand_time_stamp.to_numpy()[-padding_len:])
        padding_time_stamp = np.stack(padding_time_stamp)
        whole_time_stamp = np.concatenate(
            (time_stamps_list, padding_time_stamp), axis=1
        )
        padding_mark = get_time_mark(whole_time_stamp, 1, self.config.freq)
        return padding_mark

    def validate(self, valid_data_loader, criterion):
        config = self.config
        total_loss = []
        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for input, target, input_mark, target_mark in valid_data_loader:
            input, target, input_mark, target_mark = (
                input.to(device),
                target.to(device),
                input_mark.to(device),
                target_mark.to(device),
            )
            # decoder input
            dec_input = torch.zeros_like(target[:, -config.horizon :, :]).float()
            dec_input = (
                torch.cat([target[:, : config.label_len, :], dec_input], dim=1)
                .float()
                .to(device)
            )

            output = self.model(input, input_mark, dec_input, target_mark, device)

            if self.model_name == "UniTimeModel":
                output = output[:, config.max_backcast_len:
                                  config.max_backcast_len+config.pred_len, :]

            target = target[:, -config.horizon :, :]
            output = output[:, -config.horizon :, :]
            loss = criterion(output, target).detach().cpu().numpy()
            total_loss.append(loss)

        total_loss = np.mean(total_loss)
        self.model.train()
        return total_loss

    def forecast_fit(
        self, train_valid_data: pd.DataFrame, train_ratio_in_tv: float
    ) -> "ModelBase":
        """
        Train the model.

        :param train_data: Time series data used for training.
        :param train_ratio_in_tv: Represents the splitting ratio of the training set validation set. If it is equal to 1, it means that the validation set is not partitioned.
        :return: The fitted model object.
        """
        if train_valid_data.shape[1] == 1:
            train_drop_last = False
        else:
            train_drop_last = True
            self.multi_forecasting_hyper_param_tune(train_valid_data)


        print(
            "----------------------------------------------------------",
            self.model_name,
        )
        config = self.config
        train_data, valid_data = train_val_split(
            train_valid_data, train_ratio_in_tv, config.seq_len
        )

        self.scaler.fit(train_data.values)
        # self.lags = get_lags(self.config.freq)

        if config.norm:
            train_data = pd.DataFrame(
                self.scaler.transform(train_data.values),
                columns=train_data.columns,
                index=train_data.index,
            )

        if train_ratio_in_tv != 1:
            if config.norm:
                valid_data = pd.DataFrame(
                    self.scaler.transform(valid_data.values),
                    columns=valid_data.columns,
                    index=valid_data.index,
                )
            valid_dataset, valid_data_loader = forecasting_data_provider(
                valid_data,
                config,
                timeenc=1,
                batch_size=config.batch_size,
                shuffle=True,
                drop_last=False,
                data_info='val',
            )

        train_dataset, train_data_loader = forecasting_data_provider(
            train_data,
            config,
            timeenc=1,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=train_drop_last,
            data_info='train',
            sampling_rate=config.sampling_rate,
            sampling_strategy=config.sampling_strategy,
            sampling_basis=config.sampling_basis,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self.model_class(self.config, device)
        
        total_params = sum(
            p.numel() for p in self.model.parameters()
        ) 
        print(f"Total parameters: {total_params}")

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=config.lr)
        self.early_stopping = EarlyStopping(patience=config.patience)
        self.model.to(device)
        total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f"Total trainable parameters: {total_params}")

        if not os.path.exists("ts_benchmark/baselines/LLM/checkpoints/LLM"):
            os.makedirs("ts_benchmark/baselines/LLM/checkpoints/LLM")

        for epoch in range(config.num_epochs):
            self.model.train()
            for i, (input, target, input_mark, target_mark) in enumerate(
                train_data_loader
            ):
                optimizer.zero_grad()
                input, target, input_mark, target_mark = (
                    input.to(device),
                    target.to(device),
                    input_mark.to(device),
                    target_mark.to(device),
                )
                # decoder input
                dec_input = torch.zeros_like(target[:, -config.horizon :, :]).float()
                dec_input = (
                    torch.cat([target[:, : config.label_len, :], dec_input], dim=1)
                    .float()
                    .to(device)
                )

                output = self.model(input, input_mark, dec_input, target_mark, device)

                if self.model_name == "UniTimeModel":
                    if config.max_backcast_len == 0:
                        output = output[:, :config.pred_len, :]
                        target = target[..., :]
                    elif config.max_forecast_len == 0:
                        output = output[:, config.max_backcast_len-config.seq_len:, :]
                        target = target[..., :]
                    else:
                        output = output[:, config.max_backcast_len-config.seq_len:
                                        config.max_backcast_len+config.pred_len, :]
                        target = torch.cat((input, target[:, -config.horizon :, :]), dim=1)
                        target = target[..., :]
                else:
                    target = target[:, -config.horizon :, :]
                    output = output[:, -config.horizon :, :]
                loss = criterion(output, target)

                loss.backward()
                optimizer.step()

            self.ending = True
            if train_ratio_in_tv != 1:
                valid_loss = self.validate(valid_data_loader, criterion)
                self.early_stopping(valid_loss, self.model)
                if self.early_stopping.early_stop:
                    self.ending = False
                    print("Early Stopping")
                    path = 'ts_benchmark/baselines/LLM/checkpoints/LLM'
                    torch.save(self.model.state_dict(), path + '/' + self.model_name + '_' + self.config.dataset + '_' + str(self.config.seq_len) + '.pth')
                    break

            adjust_learning_rate(optimizer, epoch + 1, config)

        if self.ending:
            print("Ending")
            path = 'ts_benchmark/baselines/LLM/checkpoints/LLM'
            torch.save(self.model.state_dict(), path + '/' + self.model_name + '_' + self.config.dataset + '_' + str(self.config.seq_len) + '.pth')
                        

    def forecast(self, horizon: int, train: pd.DataFrame) -> np.ndarray:
        return None

    def batch_forecast(
        self, horizon: int, batch_maker: BatchMaker, **kwargs
    ) -> np.ndarray:
        """
        Make predictions by batch.

        :param horizon: The length of each prediction.
        :param batch_maker: Make batch data used for prediction.
        :return: An array of predicted results.
        """
        if hasattr(self, 'early_stopping') and self.early_stopping.check_point is not None:
            self.model.load_state_dict(self.early_stopping.check_point)
        elif self.config.get_train or self.ending:
            path = 'ts_benchmark/baselines/LLM/checkpoints/LLM/' + self.model_name + '_' + self.config.dataset + '_' + str(self.config.seq_len) + '.pth'
            self.model.load_state_dict(torch.load(path))

        if self.model is None:
            raise ValueError("Model not trained. Call the fit() function first.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()

        input_data = batch_maker.make_batch(self.config.batch_size, self.config.seq_len)
        input_np = input_data["input"]

        if self.config.norm:
            origin_shape = input_np.shape
            flattened_data = input_np.reshape((-1, input_np.shape[-1]))
            input_np = self.scaler.transform(flattened_data).reshape(origin_shape)

        input_index = input_data["time_stamps"]
        padding_len = (
            math.ceil(horizon / self.config.horizon) + 1
        ) * self.config.horizon
        all_mark = self._padding_time_stamp_mark(input_index, padding_len)

        answers = self._perform_rolling_predictions(horizon, input_np, all_mark, device)

        if self.config.norm:
            flattened_data = answers.reshape((-1, answers.shape[-1]))
            answers = self.scaler.inverse_transform(flattened_data).reshape(
                answers.shape
            )

        return answers

    def _perform_rolling_predictions(
        self,
        horizon: int,
        input_np: np.ndarray,
        all_mark: np.ndarray,
        device: torch.device,
    ) -> list:
        """
        Perform rolling predictions using the given input data and marks.

        :param horizon: Length of predictions to be made.
        :param input_np: Numpy array of input data.
        :param all_mark: Numpy array of all marks (time stamps mark).
        :param device: Device to run the model on.
        :return: List of predicted results for each prediction batch.
        """
        rolling_time = 0
        input_np, target_np, input_mark_np, target_mark_np = self._get_rolling_data(
            input_np, None, all_mark, rolling_time
        )
        with torch.no_grad():
            answers = []
            while not answers or sum(a.shape[1] for a in answers) < horizon:
                input, dec_input, input_mark, target_mark = (
                    torch.tensor(input_np, dtype=torch.float32).to(device),
                    torch.tensor(target_np, dtype=torch.float32).to(device),
                    torch.tensor(input_mark_np, dtype=torch.float32).to(device),
                    torch.tensor(target_mark_np, dtype=torch.float32).to(device),
                )
                output = self.model(input, input_mark, dec_input, target_mark, device)
                if self.model_name == "UniTimeModel":
                    output = output[:, self.config.max_backcast_len:
                                    self.config.max_backcast_len+self.config.pred_len, :]
                column_num = output.shape[-1]
                real_batch_size = output.shape[0]
                answer = (
                    output.cpu()
                    .numpy()
                    .reshape(real_batch_size, -1, column_num)[
                        :, -self.config.horizon :, :
                    ]
                )
                answers.append(answer)
                if sum(a.shape[1] for a in answers) >= horizon:
                    break
                rolling_time += 1
                output = output.cpu().numpy()[:, -self.config.horizon :, :]
                (
                    input_np,
                    target_np,
                    input_mark_np,
                    target_mark_np,
                ) = self._get_rolling_data(input_np, output, all_mark, rolling_time)

        answers = np.concatenate(answers, axis=1)
        return answers[:, -horizon:, :]

    def _get_rolling_data(
        self,
        input_np: np.ndarray,
        output: Optional[np.ndarray],
        all_mark: np.ndarray,
        rolling_time: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare rolling data based on the current rolling time.

        :param input_np: Current input data.
        :param output: Output from the model prediction.
        :param all_mark: Numpy array of all marks (time stamps mark).
        :param rolling_time: Current rolling time step.
        :return: Updated input data, target data, input marks, and target marks for rolling prediction.
        """
        if rolling_time > 0:
            input_np = np.concatenate((input_np, output), axis=1)
            input_np = input_np[:, -self.config.seq_len :, :]
        target_np = np.zeros(
            (
                input_np.shape[0],
                self.config.label_len + self.config.horizon,
                input_np.shape[2],
            )
        )
        target_np[:, : self.config.label_len, :] = input_np[
            :, -self.config.label_len :, :
        ]
        advance_len = rolling_time * self.config.horizon
        input_mark_np = all_mark[:, advance_len : self.config.seq_len + advance_len, :]
        start = self.config.seq_len - self.config.label_len + advance_len
        end = self.config.seq_len + self.config.horizon + advance_len
        target_mark_np = all_mark[
            :,
            start:end,
            :,
        ]
        return input_np, target_np, input_mark_np, target_mark_np

def generate_model_factory(
    model_name: str, model_class: type, required_args: dict
) -> Dict:
    """
    Generate model factory information for creating Transformer Adapters model adapters.

    :param model_name: Model name.
    :param model_class: Model class.
    :param required_args: The required parameters for model initialization.
    :return: A dictionary containing model factories and required parameters.
    """

    def model_factory(**kwargs) -> LLMAdapter:
        """
        Model factory, used to create TransformerAdapter model adapter objects.

        :param kwargs: Model initialization parameters.
        :return:  Model adapter object.
        """
        return LLMAdapter(model_name, model_class, **kwargs)

    return {
        "model_factory": model_factory,
        "required_hyper_params": required_args,
    }


def llm_adapter(model_info: Type[object]) -> object:
    if not isinstance(model_info, type):
        raise ValueError("the model_info does not exist")

    return generate_model_factory(
        model_name=model_info.__name__,
        model_class=model_info,
        required_args={
            "seq_len": "input_chunk_length",
            "horizon": "output_chunk_length",
            "norm": "norm",
        },
    )
