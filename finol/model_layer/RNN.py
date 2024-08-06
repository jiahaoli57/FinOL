import torch
import torch.nn as nn

from einops import rearrange
from finol.data_layer.scaler_selector import ScalerSelector
from finol.utils import load_config


class RNN(nn.Module):
    """
    Recurrent Neural Network (RNN) model for portfolio selection.

    The RNN model takes an input tensor ``x`` of shape `(batch_size, num_assets, num_features_augmented)`,
    where `num_features_augmented` represents the number of features (including any preprocessed or augmented
    features) for each asset.

    The final output of the model is a tensor of shape `(batch_size, num_assets)`, where each element
    represents the predicted score for the corresponding asset.

    :param model_args: Dictionary containing model arguments, such as the number of features.
    :param model_params: Dictionary containing model hyperparameters, such as the number of layers, the hidden size, and the dropout rate.

    .. note::

        Users can refer to this implementation and use it as a starting point for developing their own advanced RNN-based models.

    Example:
        .. code:: python
        >>> from finol.data_layer.dataset_loader import DatasetLoader
        >>> from finol.model_layer.model_selector import ModelSelector
        >>> from finol.utils import load_config, update_config, portfolio_selection
        >>>
        >>> # Configuration
        >>> config = load_config()
        >>> config["MODEL_NAME"] = "RNN"
        >>> config["MODEL_PARAMS"]["RNN"]["NUM_LAYERS"] = 1
        >>> config["MODEL_PARAMS"]["RNN"]["HIDDEN_SIZE"] = 64
        >>> config["MODEL_PARAMS"]["RNN"]["DROPOUT"] = 0.1
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

    \\
    """
    def __init__(self, model_args, model_params):
        super().__init__()
        self.config = load_config()
        self.model_args = model_args
        self.model_params = model_params

        self.rnn = nn.RNN(
            input_size=model_args["num_features_original"],
            hidden_size=model_params["HIDDEN_SIZE"],
            num_layers=model_params["NUM_LAYERS"],
            batch_first=True
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(model_params["DROPOUT"])
        self.fc = nn.Linear(model_params["HIDDEN_SIZE"], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        :param x: Input tensor of shape `(batch_size, num_assets, num_features_augmented)`.
        :return: Output tensor of shape `(batch_size, num_assets)` containing the predicted scores for each asset.
        """
        batch_size, num_assets, num_features_augmented = x.shape

        """Input Transformation"""
        x = x.view(batch_size, num_assets, self.model_args["window_size"], self.model_args["num_features_original"])
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
