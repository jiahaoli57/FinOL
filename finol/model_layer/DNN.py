import torch.nn as nn
import torch.nn.functional as F
from finol.config import *

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

        self.fc1 = nn.Linear(self.input_size, HIDDEN_SIZE)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=DROPOUT)
        self.fc2 = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x):
        batch_size, num_assets, num_features_augmented = x.shape  # n: window size; d: number of features
        # x = x.view(batch_size, num_assets, self.window_size, self.num_features_original)

        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out).squeeze(-1)
        portfolio = F.softmax(out, dim=-1)


        return portfolio