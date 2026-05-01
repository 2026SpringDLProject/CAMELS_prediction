import importlib.util
from pathlib import Path

import torch
import torch.nn as nn

try:
    from utils.tftinput import TFTInput
except ModuleNotFoundError:
    PROJECT_DIR = Path(__file__).resolve().parents[1]
    module_path = PROJECT_DIR / "utils" / "tftinput.py"
    spec = importlib.util.spec_from_file_location("camels_tftinput", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    TFTInput = module.TFTInput


class LSTMBaseline(nn.Module):
    """LSTM-only baseline that uses the same inputs and output contract as SimpleTFT.

    The baseline keeps basin/static conditioning and optional future-known decoder
    inputs, but removes variable selection, self-attention, cross-attention, and
    gated residual blocks. This makes it a useful architecture baseline under the
    same training loop, loss functions, and metrics.
    """

    def __init__(
        self,
        num_real_features,
        num_basins,
        prediction_length=1,
        num_future_known_features=0,
        num_static_features=0,
        d_model=64,
        lstm_hidden=64,
        dropout=0.1,
    ):
        super().__init__()

        self.prediction_length = prediction_length
        self.num_future_known_features = num_future_known_features
        self.num_static_features = num_static_features
        self.input_parser = TFTInput()

        self.basin_embedding = nn.Embedding(num_basins, d_model)
        self.static_attribute_encoder = (
            nn.Sequential(
                nn.Linear(num_static_features, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
            )
            if num_static_features > 0
            else None
        )

        self.input_projection = nn.Sequential(
            nn.Linear(num_real_features, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.static_to_h0 = nn.Linear(d_model, lstm_hidden)
        self.static_to_c0 = nn.Linear(d_model, lstm_hidden)
        self.encoder = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )

        self.horizon_embedding = nn.Embedding(prediction_length, lstm_hidden)
        self.static_context_proj = nn.Linear(d_model, lstm_hidden)
        self.future_known_projection = (
            nn.Sequential(
                nn.Linear(num_future_known_features, lstm_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(lstm_hidden, lstm_hidden),
            )
            if num_future_known_features > 0
            else None
        )
        self.decoder = nn.LSTM(
            input_size=lstm_hidden,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, 1),
        )

    def forward(self, x, future_known=None, static_features=None):
        x_real, basin_code = self.input_parser(x)
        basin_context = self.basin_embedding(basin_code)
        if self.static_attribute_encoder is not None and static_features is not None:
            basin_context = basin_context + self.static_attribute_encoder(static_features)

        encoder_input = self.input_projection(x_real) + basin_context.unsqueeze(1)
        h0 = self.static_to_h0(basin_context).unsqueeze(0)
        c0 = self.static_to_c0(basin_context).unsqueeze(0)
        _, (h_enc, c_enc) = self.encoder(
            encoder_input, (h0.contiguous(), c0.contiguous())
        )

        horizon_ids = torch.arange(self.prediction_length, device=x.device)
        decoder_input = (
            self.static_context_proj(basin_context).unsqueeze(1)
            + self.horizon_embedding(horizon_ids).unsqueeze(0)
        )
        if self.future_known_projection is not None and future_known is not None:
            decoder_input = decoder_input + self.future_known_projection(future_known)

        decoder_out, _ = self.decoder(
            decoder_input, (h_enc.contiguous(), c_enc.contiguous())
        )
        y_hat = self.head(decoder_out).squeeze(-1)

        if self.prediction_length == 1:
            y_hat = y_hat.squeeze(-1)

        return y_hat, {
            "variable_weights": None,
            "attention_weights": None,
            "decoder_attention_weights": None,
        }
