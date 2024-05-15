import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from finol.config import *

NUM_LAYERS = MODEL_CONFIG.get("Transformer")["NUM_LAYERS"]
NUM_HEADS = MODEL_CONFIG.get("Transformer")["NUM_HEADS"]
HIDDEN_SIZE = MODEL_CONFIG.get("Transformer")["HIDDEN_SIZE"]


class Transformer(nn.Module):
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
        self.input_size = num_features_original
        self.num_feats = num_features_original
        self.output_size = num_assets
        self.num_features_augmented = num_features_augmented
        self.num_features_original = num_features_original
        self.window_size = window_size

        self.model = nn.TransformerEncoderLayer(d_model=self.input_size, nhead=NUM_HEADS, dim_feedforward=HIDDEN_SIZE, batch_first=True)
        self.dropout = nn.Dropout(p=DROPOUT)
        self.linear = nn.Linear(self.input_size, 1)

    def forward(
            self,
            x
    ):
        batch_size = x.shape[0]

        # Input Transformation
        x = x.view(batch_size, self.num_assets, self.window_size, self.num_features_original)
        x = rearrange(x, 'b m n d -> (b m) n d')

        # Temporal Representation Extraction
        output = self.model(x)
        output = output[:, -1, :]

        # Final Scores for Assets
        output = output.view(batch_size, self.num_assets, self.num_features_original)
        output = self.dropout(output)
        output = F.relu(output)
        output = self.linear(output).squeeze(-1)
        # portfolio = F.softmax(output, dim=-1)
        return output
