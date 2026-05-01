import torch
import torch.nn as nn

# The class is used to extract the basin_code from the input data， the feature column is 
# [
#     'Dayl(s)',
#     'PRCP(mm/day)',
#     'SRAD(W/m2)',
#     'SWE(mm)',
#     'Tmax(C)',
#     'Tmin(C)',
#     'Vp(Pa)',
#     'doy_sin',
#     'doy_cos',
#     'basin_code',
# ]
# The first 9 columns are the real-valued features, and the last column is the basin code (categorical feature). The TFTInput class will separate the real-valued features and the basin code, and return them as two separate tensors. The real-valued features will be passed to the VariableSelectionNetwork, and the basin code will be passed to the embedding layer in the SimpleTFT model.

class TFTInput(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        num_features = x.size(-1)
        basin_idx = num_features - 1

        x_real = torch.cat((x[..., :basin_idx], x[..., basin_idx + 1 :]), dim=-1).float()

        # basin_code is constant within one basin window. Use the first time step as the sample label.
        basin_code = x[:, 0, basin_idx].long()

        return x_real, basin_code
