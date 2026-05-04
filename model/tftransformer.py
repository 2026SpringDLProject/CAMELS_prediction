import torch
import torch.nn as nn
from utils.tftinput import TFTInput

# The GatedResidualNetwork is a building block of the VariableSelectionNetwork.
class GatedResidualNetwork(nn.Module):
    def __init__(self, d_in, d_hidden, d_out=None, dropout=0.1):
        """
        input x: [B, T, d_in]
        output: [B, T, d_out]
        d_in: input dimension
        d_hidden: hidden dimension in the GRN
        d_out: output dimension (if None, set to d_in)
        """
        super().__init__()
        d_out = d_out or d_in
        self.fc1 = nn.Linear(d_in, d_hidden) # fully connected layer
        self.elu = nn.ELU() # activation function
        self.fc2 = nn.Linear(d_hidden, d_out) # fully connected layer
        self.dropout = nn.Dropout(dropout) # dropout layer
        # input gate, decide whether to use the transformation in a specific dimension
        self.gate = nn.Sequential(
            nn.Linear(d_out, d_out),
            nn.Sigmoid()
        )
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.norm = nn.LayerNorm(d_out)

    def forward(self, x):
        residual = self.skip(x) # skip connection
        x = self.fc1(x)
        x = self.elu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = x * self.gate(x)
        return self.norm(x + residual)


class VariableSelectionNetwork(nn.Module):
    def __init__(self, num_features, d_model, dropout=0.1):
        """ Input x: [B, T, F]
            B: batch size, T: time steps, F: number of features
            Output: [B, T, D], variable weights: [B, T, F]
            D: model dimension after feature transformation
        """
        super().__init__()
        self.num_features = num_features
        # each feature has its own GRN
        self.feature_grns = nn.ModuleList([
            GatedResidualNetwork(1, d_model, d_model, dropout) for _ in range(num_features) # (B, T, 1) -> (B, T, D)
        ])
        self.weight_grn = GatedResidualNetwork(num_features, d_model, num_features, dropout) # (B, T, F) -> (B, T, F)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, T, F = x.shape
        weights = self.softmax(self.weight_grn(x))   # [B, T, F]

        transformed = [] # A list of transformed features, each feature is [B, T, D]
        for i in range(F):
            feat_i = x[:, :, i:i+1]                           # [B, T, 1]
            transformed.append(self.feature_grns[i](feat_i))  # [B, T, D]

        transformed = torch.stack(transformed, dim=2)         # [B, T, F, D]
        weights = weights.unsqueeze(-1)                       # [B, T, F, 1]
        out = (weights * transformed).sum(dim=2)              # [B, T, D]
        return out, weights.squeeze(-1)


class SimpleTFT(nn.Module):
    def __init__(
        self,
        num_real_features,
        num_basins,
        prediction_length=1,
        d_model=64,
        lstm_hidden=64,
        n_heads=4,
        dropout=0.1,
    ):
        super().__init__()
        if prediction_length < 1:
            raise ValueError(f"prediction_length must be >= 1, got {prediction_length}")

        self.prediction_length = prediction_length
        self.input_parser = TFTInput()

        # Embedding layer for basin code [B,1] -> [B, D]
        self.basin_embedding = nn.Embedding(num_basins, d_model)

        # VSN [B, T, F] -> [B, T, D]
        self.vsn = VariableSelectionNetwork(num_real_features, d_model, dropout)

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
        # Decoder rolls the encoded context forward for prediction_length steps.
        self.decoder = nn.LSTM(
            input_size=lstm_hidden,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True
        )
        # Prediction head [B, prediction_length, H] -> [B, prediction_length, 1]
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, 1),
        )

    def forward(self, x):
        # x: [B, T, F]， where F includes both real-valued features and the basin code. Use TFTInput to separate them.
        x_real, basin_code = self.input_parser(x)          # [B, T, F_real], [B]
        # basin_code: [B]
        x, var_weights = self.vsn(x_real)                    # [B, T, D], [B, T, F_real]

        basin_context = self.basin_embedding(basin_code)     # [B, D]
        x = x + basin_context.unsqueeze(1)                   # broadcast to time axis [B, T, D]

        x, _ = self.lstm(x)                                  # [B, T, H]
        attn_out, attn_weights = self.attn(x, x, x)          # [B, T, H], [B, T, T]
        x = self.post_attn_grn(attn_out)                     # [B, T, H]

        context = x[:, -1:, :]                               # [B, 1, H]
        decoder_input = context.expand(-1, self.prediction_length, -1)
        decoder_out, _ = self.decoder(decoder_input)         # [B, prediction_length, H]
        y_hat = self.head(decoder_out).squeeze(-1)           # [B, prediction_length]

        if self.prediction_length == 1:
            y_hat = y_hat.squeeze(-1)                        # [B]

        return y_hat, {
            "variable_weights": var_weights,
            "attention_weights": attn_weights,
        }
