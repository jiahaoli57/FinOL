import time
import torch
import torch.nn as nn

from finol.data_layer.scaler_selector import ScalerSelector
from finol.utils import load_config


class DNN(nn.Module):
    """
    Class to generate predicted scores for the input assets based on the Deep Neural Network (DNN) model.

    The DNN model takes an input tensor ``x`` of shape `(batch_size, num_assets, num_features_augmented)`,
    where `num_features_augmented` represents the number of features (including any preprocessed or augmented
    features) for each asset. The model applies a series of fully connected layers to the input,
    with each layer followed by a ReLU activation and a dropout layer.

    The final output of the model is a tensor of shape `(batch_size, num_assets)`, where each element
    represents the predicted score for the corresponding asset.

    :param model_args: Dictionary containing model arguments, such as the number of features.
    :param model_params: Dictionary containing model hyper-parameters, such as the number of layers, the hidden size, and the dropout rate.

    Example:
        .. code:: python
        >>> from finol.data_layer.dataset_loader import DatasetLoader
        >>> from finol.model_layer.model_instantiator import ModelInstantiator
        >>> from finol.utils import load_config, update_config, portfolio_selection
        >>>
        >>> # Configuration
        >>> config = load_config()
        >>> config["MODEL_NAME"] = "DNN"
        >>> config["MODEL_PARAMS"]["DNN"]["NUM_LAYERS"] = 1
        >>> config["MODEL_PARAMS"]["DNN"]["HIDDEN_SIZE"] = 64
        >>> config["MODEL_PARAMS"]["DNN"]["DROPOUT"] = 0.1
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

        Users can refer to this implementation and use it as a starting point for developing their own advanced DNN-based models.

    \\
    """
    def __init__(self, model_args, model_params) -> None:
        super().__init__()
        self.config = load_config()
        self.model_args = model_args
        self.model_params = model_params

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(model_args["num_features_augmented"], model_params["HIDDEN_SIZE"]))
        for _ in range(model_params["NUM_LAYERS"]):
            self.layers.append(nn.Linear(model_params["HIDDEN_SIZE"], model_params["HIDDEN_SIZE"]))
        self.layers.append(nn.Linear(model_params["HIDDEN_SIZE"], 1))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=model_params["DROPOUT"])

        # 手动初始化模型参数
        self.init_weights()

    def init_weights(self):
        import torch.nn.init as init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用 Xavier 初始化方法
                init.xavier_uniform_(m.weight)
                init.constant_(m.bias, 0)  # 假设偏置初始化为0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        :param x: Input tensor of shape `(batch_size, num_assets, num_features_augmented)`.
        :return: Output tensor of shape `(batch_size, num_assets)` containing the predicted scores for each asset.
        """
        batch_size, num_assets, num_features_augmented = x.shape
        # x = x.view(batch_size, num_assets, window_size, num_features_original)

        """Input Transformation"""
        if self.config["SCALER"].startswith("Window"):
            x = ScalerSelector().window_normalize(x)

        out = x
        print(out)
        time.sleep(1111)
        for layer in self.layers:
            out = layer(out)
            out = self.relu(out)
            out = self.dropout(out)

        """Final Scores for Assets"""
        final_scores = out.squeeze(-1)
        return final_scores