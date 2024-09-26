import time

import torch
import torch.nn as nn

from einops import rearrange
from finol.data_layer.scaler_selector import ScalerSelector
from finol.utils import load_config


class CNN(nn.Module):
    """
    Class to generate predicted scores for the input assets based on the Convolutional Neural Network (CNN) model.

    The CNN model takes an input tensor ``x`` of shape `(batch_size, num_assets, height, width)`, where
    `height` and `width` are the dimensions of the image for each asset. The model applies a series of convolutional
    layers to each asset's image, with each layer followed by a ReLU activation and a pooling layer to reduce
    the spatial dimensions.

    The final output of the model is a tensor of shape ``(batch_size, num_assets)``, where each element
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
        >>> config["MODEL_NAME"] = "CNN"
        >>> config["MODEL_PARAMS"]["CNN"]["KERNEL_SIZE"] = 3
        >>> config["MODEL_PARAMS"]["CNN"]["STRIDE"] = 1
        >>> config["MODEL_PARAMS"]["CNN"]["HIDDEN_SIZE"] = 4
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

        Users can refer to this implementation and use it as a starting point for developing their own advanced CNN-based models.

    \\
    """
    def __init__(self, model_args, model_params):
        super().__init__()
        self.config = load_config()
        self.model_args = model_args
        self.model_params = model_params

        # Convolutional layers
        self.conv_layer_1 = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=model_params["KERNEL_SIZE"], stride=model_params["STRIDE"]),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
        )
        self.conv_layer_2 = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=model_params["KERNEL_SIZE"], stride=model_params["STRIDE"]),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
        )
        self.conv_layer_3 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=model_params["KERNEL_SIZE"], stride=model_params["STRIDE"]),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
        )

        # Calculate the size of the output after the convolutions and pooling layers
        with torch.no_grad():
            dummy_input = torch.zeros((1, 1, self.config["DATA_AUGMENTATION_CONFIG"]["IMAGE_DATA"]["SIDE_LENGTH"], self.config["DATA_AUGMENTATION_CONFIG"]["IMAGE_DATA"]["SIDE_LENGTH"]))
            x = self.conv_layer_1(dummy_input)
            x = self.conv_layer_2(x)
            x = self.conv_layer_3(x)
            conv_output_size = x.numel()  # Calculate total number of elements

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_output_size, model_params["HIDDEN_SIZE"]),
            nn.ReLU(),
            nn.Dropout(model_params["DROPOUT"]),
            nn.Linear(model_params["HIDDEN_SIZE"], 1)  # Each asset gets one score
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        :param x: Input tensor of shape `(batch_size, num_assets, height, width)`.
        :return: Output tensor of shape ``(batch_size, num_assets)`` containing the predicted scores for each asset.
        """
        batch_size, num_assets, height, width = x.size()

        """Input Transformation"""
        # Reshape the input to fit the convolutional layers
        x = x.view(-1, 1, height, width)  # [batch_size, num_assets, height, width] -> [batch_size * num_assets, height, width]

        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        x = self.conv_layer_3(x)
        x = self.fc_layers(x)

        """Final Scores for Assets"""
        final_scores = x.view(batch_size, num_assets)  # Reshape to get scores for each asset
        return final_scores
