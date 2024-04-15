import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from finol.config import *

NUM_LAYERS = MODEL_CONFIG.get("LSTM")["NUM_LAYERS"]
HIDDEN_SIZE = MODEL_CONFIG.get("LSTM")["HIDDEN_SIZE"]


class LSTM(nn.Module):
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

        self.num_assets = num_assets
        self.num_features_augmented = num_features_augmented
        self.num_features_original = num_features_original
        self.window_size = window_size

        self.lstm = nn.LSTM(input_size=num_features_original, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, batch_first=True)
        self.dropout = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(HIDDEN_SIZE, 1)


    def forward(self, x):
        batch_size = x.shape[0]

        # Input Transformation
        x = x.view(batch_size, self.num_assets, self.window_size, self.num_features_original)
        x = rearrange(x, 'b m n d -> (b m) n d')

        # Temporal Representation Extraction
        out, _ = self.lstm(x)
        out = self.dropout(out)
        out = out[:, -1, :]

        # Decision Making
        out = out.view(batch_size, self.num_assets, HIDDEN_SIZE)
        out = self.fc(out).squeeze(-1)
        # portfolio = F.softmax(out, dim=-1)
        return out