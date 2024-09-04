import torch
import torch.nn as nn

from einops import rearrange
from finol.data_layer.scaler_selector import ScalerSelector
from finol.utils import load_config


# User-defined model class
class CustomModel(nn.Module):
    """
    CustomModel as a base neural network model for portfolio selection.

    Users can extend this class and implement their desired model architecture and functionality.

    :param model_args: Dictionary containing model arguments, such as the number of features.
    :param model_params: Dictionary containing model hyper-parameters, such as the parameter1, parameter2, etc.

    Example:
        .. code:: python
        >>> from finol.data_layer.dataset_loader import DatasetLoader
        >>> from finol.model_layer.model_instantiator import ModelInstantiator
        >>> from finol.utils import load_config, update_config, portfolio_selection
        >>>
        >>> # Configuration
        >>> config = load_config()
        >>> config["MODEL_NAME"] = "CustomModel"
        >>> config["MODEL_PARAMS"]["CustomModel"]["PARAMETER1"] = 2
        >>> config["MODEL_PARAMS"]["CustomModel"]["PARAMETER1"] = 128
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

    .. warning::
        When users define their own model, besides modifying this class, they must add different parameter keys and values
        in the ``config.json`` at the location ``config["MODEL_PARAMS"]["CustomModel"]``. Similarly, if users want to implement
        automatic hyper-parameters tuning for their custom model, they also need to specify the range and type of different
        parameters at ``config["MODEL_PARAMS_SPACE"]["CustomModel"]``

    \\
    """
    def __init__(self, model_args, model_params):
        super().__init__()
        self.config = load_config()
        self.model_args = model_args
        self.model_parms = model_params
        # Define your model architecture here

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

        ...

        final_scores = x

        return final_scores