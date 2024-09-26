import torch
import torch.nn as nn

from einops import rearrange
from finol.data_layer.scaler_selector import ScalerSelector
from finol.utils import load_config


class TCN(nn.Module):
    """
    Class to generate predicted scores for the input assets based on the Temporal Convolutional Network (TCN) model.

    The TCN model takes an input tensor ``x`` of shape `(batch_size, num_assets, num_features_augmented)`,
    where `num_features_augmented` represents the number of features (including any preprocessed or augmented
    features) for each asset. The model applies a series of fully connected layers to the input,
    with each layer followed by a ReLU activation and a dropout layer.

    The TCN model takes an input tensor ``x`` of shape `(batch_size, num_assets, num_features_augmented)`,
    where `num_features_augmented` represents the number of features (including any preprocessed or augmented
    features) for each asset. The final output of the model is a tensor of shape `(batch_size, num_assets)`, where each element
    represents the predicted score for the corresponding asset.

    :param model_args: Dictionary containing model arguments, such as the number of features.
    :param model_params: Dictionary containing model hyperparameters, such as the kernel size, the hidden size, and the dropout rate.

    Example:
        .. code:: python
        >>> from finol.data_layer.dataset_loader import DatasetLoader
        >>> from finol.model_layer.model_instantiator import ModelInstantiator
        >>> from finol.utils import load_config, update_config, portfolio_selection
        >>>
        >>> # Configuration
        >>> config = load_config()
        >>> config["MODEL_NAME"] = "TCN"
        >>> config["MODEL_PARAMS"]["TCN"]["OUT_CHANNELS"] = 128
        >>> config["MODEL_PARAMS"]["TCN"]["KERNEL_SIZE"] = 3
        >>> config["MODEL_PARAMS"]["TCN"]["STRIDE"] = 1
        >>> ...
        >>> update_config(config)
        >>>
        >>> # Data Layer
        >>> load_dataset_output = DatasetLoader().load_dataset()
        >>>
        >>> # Model Layer & Optimization Layer
        >>> ...
        >>> model = ModelInstantiator(load_dataset_output).instantiate_model()
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

        Users can refer to this implementation and use it as a starting point for developing their own advanced TCN-based models.

    \\
    """
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
        self.dropout = nn.Dropout(p=model_params["DROPOUT"])

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
        x = x.transpose(1, 2)  # [batch_size * num_assets, seq_len, num_inputs] -> [batch_size * num_assets, num_inputs, seq_len]

        if self.config["SCALER"].startswith("Window"):
            x = ScalerSelector().window_normalize(x)

        """Temporal Representation Extraction"""
        out = self.net(x)
        out = out.view(batch_size, num_assets, 1).squeeze(-1)
        out = self.dropout(out)

        """Final Scores for Assets"""
        final_scores = out
        return final_scores
