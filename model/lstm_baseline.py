import torch
import torch.nn as nn
from utils.tftinput import TFTInput


# LSTM-only baseline that uses the same inputs and output contract as SimpleTFT,
# but removes variable selection, self-attention, and gated residual blocks.
class LSTMBaseline(nn.Module):
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

        # Basin embedding [B] -> [B, D]
        self.basin_embedding = nn.Embedding(num_basins, d_model)
        # Static attribute encoder [B, F_stat] -> [B, D]
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

        # Input projection [B, T, F_real] -> [B, T, D]
        self.input_projection = nn.Sequential(
            nn.Linear(num_real_features, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # Initial encoder LSTM states from basin context [B, D] -> [B, H]
        self.static_to_h0 = nn.Linear(d_model, lstm_hidden)
        self.static_to_c0 = nn.Linear(d_model, lstm_hidden)
        # Encoder LSTM [B, T, D] -> [B, T, H], final (h, c): [1, B, H]
        self.encoder = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )

        # Horizon position embedding [P] -> [P, H]
        self.horizon_embedding = nn.Embedding(prediction_length, lstm_hidden)
        # Static context to decoder input [B, D] -> [B, H]
        self.static_context_proj = nn.Linear(d_model, lstm_hidden)
        # Future-known projection [B, P, F_fut] -> [B, P, H]
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
        # Decoder LSTM [B, P, H] -> [B, P, H]
        self.decoder = nn.LSTM(
            input_size=lstm_hidden,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )
        # Prediction head [B, P, H] -> [B, P, 1]
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, 1),
        )

    def forward(self, x, future_known=None, static_features=None):
        # x: [B, T, F], future_known: [B, P, F_fut] or None, static_features: [B, F_stat] or None
        x_real, basin_code = self.input_parser(x)            # [B, T, F_real], [B]
        basin_context = self.basin_embedding(basin_code)     # [B, D]
        if self.static_attribute_encoder is not None and static_features is not None:
            basin_context = basin_context + self.static_attribute_encoder(static_features)  # [B, D]

        encoder_input = self.input_projection(x_real) + basin_context.unsqueeze(1)  # [B, T, D]
        h0 = self.static_to_h0(basin_context).unsqueeze(0)   # [1, B, H]
        c0 = self.static_to_c0(basin_context).unsqueeze(0)   # [1, B, H]
        _, (h_enc, c_enc) = self.encoder(
            encoder_input, (h0.contiguous(), c0.contiguous())
        )                                                    # h_enc, c_enc: [1, B, H]

        horizon_ids = torch.arange(self.prediction_length, device=x.device)  # [P]
        decoder_input = (
            self.static_context_proj(basin_context).unsqueeze(1)             # [B, 1, H]
            + self.horizon_embedding(horizon_ids).unsqueeze(0)               # [1, P, H]
        )                                                                    # [B, P, H]
        if self.future_known_projection is not None and future_known is not None:
            decoder_input = decoder_input + self.future_known_projection(future_known)  # [B, P, H]

        decoder_out, _ = self.decoder(
            decoder_input, (h_enc.contiguous(), c_enc.contiguous())
        )                                                    # [B, P, H]
        y_hat = self.head(decoder_out).squeeze(-1)           # [B, P]

        if self.prediction_length == 1:
            y_hat = y_hat.squeeze(-1)                        # [B]

        return y_hat, {
            "variable_weights": None,
            "attention_weights": None,
            "decoder_attention_weights": None,
        }