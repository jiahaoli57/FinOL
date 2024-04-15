import torch.nn as nn
import torch.nn.functional as F
from finol.config import *

NUM_LAYERS = MODEL_CONFIG.get("DNN")["NUM_LAYERS"]
HIDDEN_SIZE = MODEL_CONFIG.get("DNN")["HIDDEN_SIZE"]


class DNN(nn.Module):
    def __init__(
            self,
            *,
            num_assets,
            num_features_augmented,
            num_features_original,
            window_size,
            **kwargs
    ):
        super().__init__()
        self.input_size = num_features_augmented
        self.num_feats = num_features_original
        self.output_size = num_assets
        self.num_features_augmented = num_features_augmented
        self.num_features_original = num_features_original
        self.window_size = window_size

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.num_features_augmented, HIDDEN_SIZE))

        for _ in range(NUM_LAYERS):
            self.layers.append(nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE))

        self.layers.append(nn.Linear(HIDDEN_SIZE, 1))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=DROPOUT)

    def forward(self, x):
        # batch_size, num_assets, num_features_augmented = x.shape  # n: window size; d: number of features
        # x = x.view(batch_size, num_assets, self.window_size, self.num_features_original)

        out = x
        for layer in self.layers:
            out = layer(out)
            out = self.relu(out)
            out = self.dropout(out)

        out = out.squeeze(-1)
        # portfolio = F.softmax(out, dim=-1)
        return out