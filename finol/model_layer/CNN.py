import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from finol.config import *

OUT_CHANNELS = MODEL_CONFIG.get("CNN")["OUT_CHANNELS"]
KERNEL_SIZE = MODEL_CONFIG.get("CNN")["KERNEL_SIZE"]
STRIDE = MODEL_CONFIG.get("CNN")["STRIDE"]
HIDDEN_SIZE = MODEL_CONFIG.get("CNN")["HIDDEN_SIZE"]


class CNN(nn.Module):
    def __init__(
            self,
            *,
            num_assets,
            num_features_augmented,
            num_features_original,
            window_size,
            **kwargs
    ):
        super(CNN, self).__init__()
        self.num_assets = num_assets
        self.num_features_augmented = num_features_augmented
        self.num_features_original = num_features_original
        self.window_size = window_size

        self.net = nn.Sequential(
            nn.Conv1d(num_features_original, OUT_CHANNELS, KERNEL_SIZE, STRIDE),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(OUT_CHANNELS, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, 1),
        )

    def forward(self, x):
        batch_size = x.shape[0]

        # # Input Transformation
        x = x.view(batch_size, self.num_assets, self.window_size, self.num_features_original)
        x = rearrange(x, 'b m n d -> (b m) n d')
        x = x.transpose(1, 2)  # [batch_size * num_assets, seq_len, num_inputs] -> [batch_size * num_assets, num_inputs, seq_len]

        # Temporal Representation Extraction
        out = self.net(x)
        out = out.view(batch_size, self.num_assets, 1).squeeze(-1)

        # Decision Making
        portfolio = F.softmax(out, dim=-1)
        return portfolio
