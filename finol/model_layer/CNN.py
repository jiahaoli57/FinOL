import torch.nn as nn

from einops import rearrange
from finol.data_layer.scaler_selector import ScalerSelector
from finol.utils import load_config


class CNN(nn.Module):
    def __init__(self, model_args, model_params):
        super().__init__()
        self.config = load_config()
        self.model_args = model_args
        self.model_params = model_params

        self.net = nn.Sequential(
            nn.Conv1d(model_args["num_features_original"], model_params["OUT_CHANNELS"], model_params["KERNEL_SIZE"], model_params["STRIDE"]),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(model_params["OUT_CHANNELS"], model_params["HIDDEN_SIZE"]),
            nn.ReLU(),
            nn.Linear(model_params["HIDDEN_SIZE"], 1),
        )

    def forward(self, x):
        batch_size, num_assets, num_features_augmented = x.shape

        """Input Transformation"""
        x = x.view(batch_size, num_assets, self.model_args["window_size"], self.model_args["num_features_original"])
        x = rearrange(x, "b m n d -> (b m) n d")
        x = x.transpose(1, 2)  # [batch_size * num_assets, seq_len, num_inputs] -> [batch_size * num_assets, num_inputs, seq_len]

        if self.config["SCALER"].startswith("Window"):
            x = ScalerSelector().window_normalize(x)

        """Temporal Representation Extraction"""
        out = self.net(x)
        out = out.view(batch_size, num_assets, 1).squeeze(-1)

        """Final Scores for Assets"""
        final_scores = out
        return final_scores
