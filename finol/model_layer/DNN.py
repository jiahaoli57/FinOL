import time

import torch.nn as nn

from finol.data_layer.scaler_selector import ScalerSelector
from finol.utils import load_config


class DNN(nn.Module):
    def __init__(self, model_args, model_params):
        super().__init__()
        self.config = load_config()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(model_args["num_features_augmented"], model_params["HIDDEN_SIZE"]))
        for _ in range(model_params["NUM_LAYERS"]):
            self.layers.append(nn.Linear(model_params["HIDDEN_SIZE"], model_params["HIDDEN_SIZE"]))
        self.layers.append(nn.Linear(model_params["HIDDEN_SIZE"], 1))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=model_params["DROPOUT"])

    def forward(self, x):
        batch_size, num_assets, num_features_augmented = x.shape
        # x = x.view(batch_size, num_assets, window_size, num_features_original)

        """Input Transformation"""
        if self.config["SCALER"].startswith("Window"):
            x = ScalerSelector().window_normalize(x)

        out = x
        for layer in self.layers:
            out = layer(out)
            out = self.relu(out)
            out = self.dropout(out)

        """Final Scores for Assets"""
        final_scores = out.squeeze(-1)
        return final_scores