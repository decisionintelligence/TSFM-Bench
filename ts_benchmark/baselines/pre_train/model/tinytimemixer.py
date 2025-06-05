
import pandas as pd
import torch
import yaml
from torch import nn
import torch.nn.functional as F
from ts_benchmark.baselines.pre_train.submodules.TinyTimeMixer.modeling_tinytimemixer import TinyTimeMixerForPrediction
from pandas.tseries.frequencies import to_offset

DEFAULT_FREQUENCY_MAPPING = {
    "oov": 0,
    "min": 4,  # minutely
    "2min": 2,
    "5min": 3,
    "10min": 4,
    "15min": 5,
    "30min": 6,
    "h": 7,  # hourly
    "d": 8,  # daily
    "W": 9,  # weekly
}

class TinyTimeMixer(nn.Module):

    def __init__(
        self,
        config,
        **kwargs,
    ):
        super().__init__()
        self.context_length = config.seq_len
        self.prediction_length = config.pred_len
        self.target_dim = config.target_dim
        self.freq = config.freq
        self.dataset = config.dataset
        self.frequency_mapping = DEFAULT_FREQUENCY_MAPPING
        self.token = self.get_frequency_token(self.freq)
        # self.no_training = True
        print(self.freq)
        print(self.token)

        with open("ts_benchmark/baselines/pre_train/checkpoints/ttm.yaml", "r") as file:
            model_revisions = yaml.safe_load(file)

        ttm_model_revision = model_revisions["research-use-models"][
            f"r2-{self.context_length}-{self.prediction_length}-freq"
        ]["revision"]

        self.model = TinyTimeMixerForPrediction.from_pretrained(
            "ibm-research/ttm-research-r2",
            revision=ttm_model_revision,
        )

        
    def get_frequency_token(self, token_name: str):
        token = self.frequency_mapping.get(token_name, None)
        if token is not None:
            return token

        # try to map as a frequency string
        try:
            token_name_offs = to_offset(token_name).freqstr
            token = self.frequency_mapping.get(token_name_offs, None)
            if token is not None:
                return token
        except ValueError:
            # lastly try to map the timedelta to a frequency string
            token_name_td = pd._libs.tslibs.timedeltas.Timedelta(token_name)
            token_name_offs = to_offset(token_name_td).freqstr
            token = self.frequency_mapping.get(token_name_offs, None)
            if token is not None:
                return token

        token = self.frequency_mapping["oov"]

        return token
    
    def forward(self, inputs, dec_inp, x_mark_enc, x_mark_dec, device=None, num_samples=None): 
        B, C, K = inputs.shape 
        freq_token = torch.full((B,), self.token).to(device)
        point_forecast = self.model(past_values=inputs, freq_token=freq_token).prediction_outputs
        return point_forecast
