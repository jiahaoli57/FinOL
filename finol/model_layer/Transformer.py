import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from finol.data_layer.scaler_selector import ScalerSelector
from finol.utils import load_config


class Transformer(nn.Module):
    """
    Transformer model for portfolio selection.

    The Transformer model takes an input tensor ``x`` of shape `(batch_size, num_assets, num_features_augmented)`,
    where `num_features_augmented` represents the number of features (including any preprocessed or augmented
    features) for each asset.

    The final output of the model is a tensor of shape `(batch_size, num_assets)`, where each element
    represents the predicted score for the corresponding asset.

    :param model_args: Dictionary containing model arguments, such as the number of features.
    :param model_params: Dictionary containing model hyperparameters, such as the number of layers, the number of heads, and the dropout rate.

    Example:
        .. code:: python
        >>> from finol.data_layer.dataset_loader import DatasetLoader
        >>> from finol.model_layer.model_selector import ModelSelector
        >>> from finol.utils import load_config, update_config, portfolio_selection
        >>>
        >>> # Configuration
        >>> config = load_config()
        >>> config["MODEL_NAME"] = "Transformer"
        >>> config["MODEL_PARAMS"]["Transformer"]["NUM_LAYERS"] = 2
        >>> config["MODEL_PARAMS"]["Transformer"]["DIM_EMBEDDING"] = 256
        >>> config["MODEL_PARAMS"]["Transformer"]["DIM_FEEDFORWARD"] = 32
        >>> ...
        >>> update_config(config)
        >>>
        >>> # Data Layer
        >>> load_dataset_output = DatasetLoader().load_dataset()
        >>>
        >>> # Model Layer & Optimization Layer
        >>> ...
        >>> model = ModelSelector(load_dataset_output).select_model()
        >>> print(f"model: {model}")
        >>> ...
        >>> train_loader = load_dataset_output["train_loader"]
        >>> for i, data in enumerate(train_loader, 1):
        ...     x_data, label = data
        ...     final_scores = model(x_data.float())
        ...     portfolio = portfolio_selection(final_scores)
        ...     print(f"batch {i} input shape: {x_data.shape}")
        ...     print(f"batch {i} label shape: {label.shape}")
        ...     print(f"batch {i} output shape: {portfolio.shape}")
        ...     print("-"*50)

    .. note::

        Users can refer to this implementation and use it as a starting point for developing their own advanced Transformer-based models.

    \\
    """
    def __init__(self, model_args, model_params):
        super().__init__()
        self.config = load_config()
        self.model_args = model_args
        self.model_params = model_params

        self.token_emb = nn.Linear(model_args["num_features_original"], model_params["DIM_EMBEDDING"])
        self.pos_emb = nn.Embedding(model_args["window_size"], model_params["DIM_EMBEDDING"])
        self.transformer_encoder = nn.TransformerEncoderLayer(model_params["DIM_EMBEDDING"], nhead=model_params["NUM_HEADS"], dim_feedforward=model_params["DIM_FEEDFORWARD"], batch_first=True)
        self.dropout = nn.Dropout(p=model_params["DROPOUT"])
        self.fc = nn.Linear(model_params["DIM_EMBEDDING"], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        :param x: Input tensor of shape `(batch_size, num_assets, num_features_augmented)`.
        :return: Output tensor of shape `(batch_size, num_assets)` containing the predicted scores for each asset.
        """
        batch_size, num_assets, num_features_augmented = x.shape
        device = x.device

        """Input Transformation"""
        x = x.view(batch_size, num_assets, self.model_args["window_size"], self.model_args["num_features_original"])
        x = rearrange(x, "b m n d -> (b m) n d")
        if self.config["SCALER"].startswith("Window"):
            x = ScalerSelector().window_normalize(x)

        """Temporal Representation Extraction"""
        x = self.token_emb(x)  # [batch_size * num_assets, window_size, num_features_original] -> [batch_size * num_assets, window_size, DIM_EMBEDDING]
        pos_emb = self.pos_emb(torch.arange(self.model_args["window_size"], device=device))
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
