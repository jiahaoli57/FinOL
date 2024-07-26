import torch.nn as nn

from einops import rearrange
from finol.data_layer.scaler_selector import ScalerSelector
from finol.utils import load_config


class RNN(nn.Module):
    def __init__(self, model_args, model_params):
        super().__init__()
        self.config = load_config()
        self.model_args = model_args
        self.model_params = model_params

        self.rnn = nn.RNN(input_size=model_args["NUM_FEATURES_ORIGINAL"], hidden_size=model_params["HIDDEN_SIZE"],
                          num_layers=model_params["NUM_LAYERS"], batch_first=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(model_params["DROPOUT"])
        self.fc = nn.Linear(model_params["HIDDEN_SIZE"], 1)

    def forward(self, x):
        batch_size, num_assets, num_features_augmented = x.shape

        """Input Transformation"""
        x = x.view(batch_size, num_assets, self.model_args["WINDOW_SIZE"], self.model_args["NUM_FEATURES_ORIGINAL"])
        x = rearrange(x, "b m n d -> (b m) n d")
        if self.config["SCALER"].startswith("Window"):
            x = ScalerSelector().window_normalize(x)

        """Temporal Representation Extraction"""
        out, _ = self.rnn(x)
        out = self.dropout(out)
        out = out[:, -1, :]

        """Final Scores for Assets"""
        out = out.view(batch_size, num_assets, self.model_params["HIDDEN_SIZE"])
        final_scores = self.fc(out).squeeze(-1)
        return final_scores
