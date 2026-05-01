import importlib.util
import sys
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

# The GatedResidualNetwork is a building block of the VariableSelectionNetwork.
class GatedResidualNetwork(nn.Module):
    def __init__(self, d_in, d_hidden, d_out=None, dropout=0.1, context_dim=None):
        """
        input x: [B, T, d_in] or [B, d_in]
        context (optional): [B, context_dim] static context — broadcast over time.
        output: same time/batch shape as x with last dim d_out.
        """
        super().__init__()
        d_out = d_out or d_in
        self.fc1 = nn.Linear(d_in, d_hidden)
        # Static context projection: a linear layer
        self.context_proj = (
            nn.Linear(context_dim, d_hidden, bias=False) if context_dim is not None else None
        )
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Sequential(
            nn.Linear(d_out, d_out),
            nn.Sigmoid()
        )
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.norm = nn.LayerNorm(d_out)

    def forward(self, x, context=None):
        residual = self.skip(x)
        h = self.fc1(x)
        if self.context_proj is not None:
            ctx = self.context_proj(context)
            if h.ndim == 3 and ctx.ndim == 2:
                ctx = ctx.unsqueeze(1)  # broadcast over time
            h = h + ctx
        h = self.elu(h)
        h = self.fc2(h)
        h = self.dropout(h)
        h = h * self.gate(h)
        return self.norm(h + residual)


class FeaturewiseGatedResidualNetwork(nn.Module):
    def __init__(self, num_features, d_hidden, d_out, dropout=0.1, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.d_hidden = d_hidden
        self.d_out = d_out
        self.eps = eps

        self.fc1_weight = nn.Parameter(torch.empty(num_features, 1, d_hidden))
        self.fc1_bias = nn.Parameter(torch.empty(num_features, d_hidden))
        self.fc2_weight = nn.Parameter(torch.empty(num_features, d_hidden, d_out))
        self.fc2_bias = nn.Parameter(torch.empty(num_features, d_out))
        self.gate_weight = nn.Parameter(torch.empty(num_features, d_out, d_out))
        self.gate_bias = nn.Parameter(torch.empty(num_features, d_out))
        self.skip_weight = nn.Parameter(torch.empty(num_features, 1, d_out))
        self.skip_bias = nn.Parameter(torch.empty(num_features, d_out))
        self.norm_weight = nn.Parameter(torch.ones(num_features, d_out))
        self.norm_bias = nn.Parameter(torch.zeros(num_features, d_out))

        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for weight in (
            self.fc1_weight,
            self.fc2_weight,
            self.gate_weight,
            self.skip_weight,
        ):
            nn.init.xavier_uniform_(weight)

        for bias in (
            self.fc1_bias,
            self.fc2_bias,
            self.gate_bias,
            self.skip_bias,
        ):
            nn.init.zeros_(bias)

    def forward(self, x):
        x = x.unsqueeze(-1)                                             # [B, T, F, 1]
        residual = torch.einsum("btfi,fid->btfd", x, self.skip_weight) + self.skip_bias

        hidden = torch.einsum("btfi,fih->btfh", x, self.fc1_weight) + self.fc1_bias
        hidden = self.elu(hidden)
        hidden = torch.einsum("btfh,fhd->btfd", hidden, self.fc2_weight) + self.fc2_bias
        hidden = self.dropout(hidden)

        gate = torch.einsum("btfd,fde->btfe", hidden, self.gate_weight) + self.gate_bias
        hidden = hidden * torch.sigmoid(gate)

        hidden = hidden + residual
        mean = hidden.mean(dim=-1, keepdim=True)
        var = hidden.var(dim=-1, keepdim=True, unbiased=False)
        hidden = (hidden - mean) / torch.sqrt(var + self.eps)
        return hidden * self.norm_weight + self.norm_bias


class VariableSelectionNetwork(nn.Module):
    def __init__(self, num_features, d_model, dropout=0.1, context_dim=None):
        """ Input x: [B, T, F]
            context (optional): [B, context_dim] static context for selection weights.
            Output: [B, T, D], variable weights: [B, T, F]
        """
        super().__init__()
        self.num_features = num_features
        # Each feature keeps its own GRN parameters, but the computation is batched across F.
        self.feature_grn = FeaturewiseGatedResidualNetwork(num_features, d_model, d_model, dropout)
        # Selection-weight GRN reads raw features and (optionally) static context, per TFT paper.
        self.weight_grn = GatedResidualNetwork(
            num_features, d_model, num_features, dropout, context_dim=context_dim
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, context=None):
        B, T, F = x.shape
        weights = self.softmax(self.weight_grn(x, context=context))   # [B, T, F]

        transformed = self.feature_grn(x)                     # [B, T, F, D]
        weights = weights.unsqueeze(-1)                       # [B, T, F, 1]
        out = (weights * transformed).sum(dim=2)              # [B, T, D]
        return out, weights.squeeze(-1)


class SimpleTFT(nn.Module):
    def __init__(
        self,
        num_real_features,
        num_basins,
        prediction_length=1,
        num_future_known_features=0,
        num_static_features=0,
        d_model=64,
        lstm_hidden=64,
        n_heads=4,
        dropout=0.1,
    ):
        super().__init__()

        self.prediction_length = prediction_length
        self.num_future_known_features = num_future_known_features
        self.num_static_features = num_static_features
        self.input_parser = TFTInput()

        # Embedding layer for basin code [B,1] -> [B, D]
        self.basin_embedding = nn.Embedding(num_basins, d_model)
        self.static_attribute_encoder = (
            GatedResidualNetwork(num_static_features, d_model, d_model, dropout)
            if num_static_features > 0
            else None
        )

        # Encoder: VSN + LSTM + Attention
        # VSN [B, T, F] -> [B, T, D], conditioned on basin_context (TFT-style).
        self.vsn = VariableSelectionNetwork(
            num_real_features, d_model, dropout, context_dim=d_model
        )

        # Project static basin context to LSTM (h0, c0) so basin identity persists in the recurrence.
        self.static_to_h0 = nn.Linear(d_model, lstm_hidden)
        self.static_to_c0 = nn.Linear(d_model, lstm_hidden)

        # LSTM layer [B, T, D] -> [B, T, H]
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True
        )
        # Multi-head attention layer [B, T, H] -> [B, T, H]
        self.attn = nn.MultiheadAttention(
            embed_dim=lstm_hidden,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        # Post-attention GRN [B, T, H] -> [B, T, H]
        self.post_attn_grn = GatedResidualNetwork(lstm_hidden, d_model, lstm_hidden, dropout) 

        # Decoder: LSTM + Prediction head
        # Horizon embedding gives each forecast step its own decoder query.
        self.decoder_horizon_embedding = nn.Embedding(prediction_length, lstm_hidden)
        self.static_context_proj = nn.Linear(d_model, lstm_hidden)
        self.future_known_grn = (
            GatedResidualNetwork(num_future_known_features, d_model, lstm_hidden, dropout)
            if num_future_known_features > 0
            else None
        )
        # Decoder LSTM [B, prediction_length, H] -> [B, prediction_length, H]
        self.decoder = nn.LSTM(
            input_size=lstm_hidden,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True
        )
        # Decoder cross-attention [B, prediction_length, H] x [B, T, H] -> [B, prediction_length, H]
        self.decoder_cross_attn = nn.MultiheadAttention(
            embed_dim=lstm_hidden,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.post_decoder_grn = GatedResidualNetwork(lstm_hidden, d_model, lstm_hidden, dropout)
        # Prediction head [B, prediction_length, H] -> [B, prediction_length, 1]
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, 1),
        )

    def forward(self, x, future_known=None, static_features=None):
        # x: [B, T, F]， where F includes both real-valued features and the basin code. Use TFTInput to separate them.
        x_real, basin_code = self.input_parser(x)          # [B, T, F_real], [B]
        # basin_code: [B]

        basin_context = self.basin_embedding(basin_code)     # [B, D]
        if self.static_attribute_encoder is not None:
            basin_context = basin_context + self.static_attribute_encoder(static_features)

        # VSN now takes the static context for selection-weight conditioning (TFT-style).
        x, var_weights = self.vsn(x_real, context=basin_context)  # [B, T, D], [B, T, F_real]
        x = x + basin_context.unsqueeze(1)                   # additive identity injection [B, T, D]

        h0 = self.static_to_h0(basin_context).unsqueeze(0)   # [1, B, H]
        c0 = self.static_to_c0(basin_context).unsqueeze(0)   # [1, B, H]
        x, (h_enc, c_enc) = self.lstm(x, (h0.contiguous(), c0.contiguous()))  # [B, T, H], ([1,B,H],[1,B,H])
        attn_out, attn_weights = self.attn(x, x, x)          # [B, T, H], [B, T, T]
        # Keep the encoder state on the residual path after self-attention.
        encoder_memory = self.post_attn_grn(x + attn_out)    # [B, T, H]

        static_context = self.static_context_proj(basin_context)  # [B, H]
        horizon_ids = torch.arange(self.prediction_length, device=x.device)
        decoder_input = (
            static_context.unsqueeze(1)
            + self.decoder_horizon_embedding(horizon_ids).unsqueeze(0)
        )
        if self.future_known_grn is not None:
            decoder_input = decoder_input + self.future_known_grn(future_known)

        # Hand encoder's final hidden/cell to the decoder LSTM (sequence-to-sequence handoff).
        decoder_out, _ = self.decoder(
            decoder_input, (h_enc.contiguous(), c_enc.contiguous())
        )                                                    # [B, prediction_length, H]
        cross_attn_out, decoder_attn_weights = self.decoder_cross_attn(
            decoder_out,
            encoder_memory,
            encoder_memory,
        )
        decoder_out = self.post_decoder_grn(decoder_out + cross_attn_out)
        y_hat = self.head(decoder_out).squeeze(-1)           # [B, prediction_length]

        if self.prediction_length == 1:
            y_hat = y_hat.squeeze(-1)                        # [B]

        return y_hat, {
            "variable_weights": var_weights,
            "attention_weights": attn_weights,
            "decoder_attention_weights": decoder_attn_weights,
        }

