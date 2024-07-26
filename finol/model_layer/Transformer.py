import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from finol.data_layer.scaler_selector import ScalerSelector
from finol.utils import load_config


class Transformer(nn.Module):
    def __init__(self, model_args, model_params):
        super().__init__()
        self.config = load_config()
        self.model_args = model_args
        self.model_params = model_params

        self.token_emb = nn.Linear(model_args["NUM_FEATURES_ORIGINAL"], model_params["DIM_EMBEDDING"])
        self.pos_emb = nn.Embedding(model_args["WINDOW_SIZE"], model_params["DIM_EMBEDDING"])
        self.transformer_encoder = nn.TransformerEncoderLayer(model_params["DIM_EMBEDDING"], nhead=model_params["NUM_HEADS"], dim_feedforward=model_params["DIM_FEEDFORWARD"], batch_first=True)
        self.dropout = nn.Dropout(p=model_params["DROPOUT"])
        self.fc = nn.Linear(model_params["DIM_EMBEDDING"], 1)

    def forward(self, x):
        batch_size, num_assets, num_features_augmented = x.shape
        DEVICE = x.device

        """Input Transformation"""
        x = x.view(batch_size, num_assets, self.model_args["WINDOW_SIZE"], self.model_args["NUM_FEATURES_ORIGINAL"])
        x = rearrange(x, "b m n d -> (b m) n d")
        if self.config["SCALER"].startswith("Window"):
            x = ScalerSelector().window_normalize(x)

        """Temporal Representation Extraction"""
        x = self.token_emb(x)  # [batch_size * num_assets, window_size, num_features_original] -> [batch_size * num_assets, window_size, DIM_EMBEDDING]
        pos_emb = self.pos_emb(torch.arange(self.model_args["WINDOW_SIZE"], device=DEVICE))
        pos_emb = rearrange(pos_emb, "n d -> () n d")
        x = x + pos_emb

        output = self.transformer_encoder(x)
        output = output[:, -1, :]

        """Final Scores for Assets"""
        output = output.view(batch_size, num_assets, self.model_params["DIM_EMBEDDING"])
        output = self.dropout(output)
        output = F.relu(output)
        final_scores = self.fc(output).squeeze(-1)
        return final_scores
