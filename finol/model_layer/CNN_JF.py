import torch
import torch.nn as nn

from einops import rearrange
from finol.data_layer.scaler_selector import ScalerSelector
from finol.utils import load_config


class CNN_JF(nn.Module):
    """
    Class to generate predicted scores for the input assets based on the CNN-JF model.

    The CNN-JF model is a CNN-based model for asset scoring and portfolio selection. It leverages CNN to analyze
    historical stock price data represented as images.

    The CNN-JF model takes an input tensor ``x`` of shape `(batch_size, num_assets, height, width)`, where
    `height` and `width` are the dimensions of the image for each asset. The model applies a series of convolutional
    layers to each asset's image, with each layer followed by a leaky ReLU activation and a pooling layer to reduce
    the spatial dimensions.

    The final output of the model is a tensor of shape ``(batch_size, num_assets)``, where each element
    represents the predicted score for the corresponding asset.

    For more details, please refer to the paper `(Re-)Imag(in)ing Price Trends <https://onlinelibrary.wiley.com/doi/epdf/10.1111/jofi.13268>`__.

    .. table:: Hyper-parameters of (Re-)Imag(in)ing Price Trends.
        :class: ghost

        +----------------------+--------+-------------------+--------+
        | Hyper-parameter      | Choice | Hyper-parameter   | Choice |
        +======================+========+===================+========+
        | Kernel Size Height   | 5      | Kernel Size Width | 3      |
        +----------------------+--------+-------------------+--------+
        | Stride Height        | 3      | Stride Width      | 1      |
        +----------------------+--------+-------------------+--------+
        | Dilation Height      | 2      | Dilation Width    | 1      |
        +----------------------+--------+-------------------+--------+
        | Padding Height       | 12     | Padding Width     | 1      |
        +----------------------+--------+-------------------+--------+
        | Dropout Rate         | 0.5    |                   |        |
        +----------------------+--------+-------------------+--------+

    :param model_args: Dictionary containing model arguments, such as the number of features.
    :param model_params: Dictionary containing model hyperparameters, such as the kernel size height, the kernel size width, and the stride height.

    Example:
        .. code:: python
        >>> from finol.data_layer.dataset_loader import DatasetLoader
        >>> from finol.model_layer.model_instantiator import ModelInstantiator
        >>> from finol.utils import load_config, update_config, portfolio_selection
        >>>
        >>> # Configuration
        >>> config = load_config()
        >>> config["MODEL_NAME"] = "CNN_JF"
        >>> config["MODEL_PARAMS"]["CNN_JF"]["KERNEL_SIZE_HEIGHT"] = 5
        >>> config["MODEL_PARAMS"]["CNN_JF"]["KERNEL_SIZE_WIDTH"] = 3
        >>> config["MODEL_PARAMS"]["CNN_JF"]["STRIDE_HEIGHT"] = 3
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
            nn.Conv2d(1, 64,
                      kernel_size=(model_params["KERNEL_SIZE_HEIGHT"], model_params["KERNEL_SIZE_WIDTH"]),
                      stride=(model_params["STRIDE_HEIGHT"], model_params["STRIDE_WIDTH"]),
                      dilation=(model_params["DILATION_HEIGHT"], model_params["DILATION_WIDTH"]),
                      padding=(model_params["PADDING_HEIGHT"], model_params["PADDING_WIDTH"])),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d((2, 1), stride=(2, 1)),
        )
        self.conv_layer_2 = nn.Sequential(
            nn.Conv2d(64, 128,
                      kernel_size=(model_params["KERNEL_SIZE_HEIGHT"], model_params["KERNEL_SIZE_WIDTH"]),
                      stride=(model_params["STRIDE_HEIGHT"], model_params["STRIDE_WIDTH"]),
                      dilation=(model_params["DILATION_HEIGHT"], model_params["DILATION_WIDTH"]),
                      padding=(model_params["PADDING_HEIGHT"], model_params["PADDING_WIDTH"])),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d((2, 1), stride=(2, 1)),
        )
        # self.conv_layer_3 = nn.Sequential(
        #     nn.Conv2d(128, 256,
        #               kernel_size=(model_params["KERNEL_SIZE_HEIGHT"], model_params["KERNEL_SIZE_WIDTH"]),
        #               stride=(model_params["STRIDE_HEIGHT"], model_params["STRIDE_WIDTH"]),
        #               dilation=(model_params["DILATION_HEIGHT"], model_params["DILATION_WIDTH"]),
        #               padding=(model_params["PADDING_HEIGHT"], model_params["PADDING_WIDTH"])),
        #     nn.BatchNorm2d(256),
        #     nn.LeakyReLU(negative_slope=0.01, inplace=True),
        #     nn.MaxPool2d((2, 1), stride=(2, 1)),
        # )

        # Calculate the size of the output after the convolutions and pooling layers
        with torch.no_grad():
            dummy_input = torch.zeros((1, 1, self.config["DATA_AUGMENTATION_CONFIG"]["IMAGE_DATA"]["SIDE_LENGTH"], self.config["DATA_AUGMENTATION_CONFIG"]["IMAGE_DATA"]["SIDE_LENGTH"]))
            x = self.conv_layer_1(dummy_input)
            x = self.conv_layer_2(x)
            # x = self.conv_layer_3(x)
            conv_output_size = x.numel()  # Calculate total number of elements

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Dropout(p=model_params["DROPOUT"]),  # 0.5
            nn.Flatten(),
            nn.Linear(conv_output_size, 1)  # Each asset gets one score
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
        # x = self.conv_layer_3(x)
        x = self.fc_layers(x)

        """Final Scores for Assets"""
        final_scores = x.view(batch_size, num_assets)  # Reshape to get scores for each asset
        return final_scores
