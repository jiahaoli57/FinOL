import time

import torch
import torch.nn.functional as F

from torch import nn
from einops import rearrange
from finol.data_layer.scaler_selector import ScalerSelector
from finol.utils import load_config


class LSTM_HA(nn.Module):
    """
    This class implements the Long Short-Term Memory with History state Attention (LSTM-HA).

    For more details, please refer to the papers `AlphaStock: A Buying-Winners-and-Selling-Losers Investment
    Strategy using Interpretable Deep Reinforcement Attention Networks <https://dl.acm.org/doi/abs/10.1145/3292500.3330647>`__
    and `Attention is all you need <https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html>`__
    """
    def __init__(self, model_args, model_params):
        super().__init__()
        self.lstm = torch.nn.LSTM(model_args["num_features_original"], model_params["HIDDEN_SIZE"],
                                  num_layers=model_params["NUM_LAYERS"], batch_first=True)
        self.dropout = nn.Dropout(model_params["DROPOUT"])
        self.linear_1 = torch.nn.Linear(model_params["HIDDEN_SIZE"], model_params["HIDDEN_SIZE"])
        self.linear_2 = torch.nn.Linear(model_params["HIDDEN_SIZE"], model_params["HIDDEN_SIZE"])
        self.w_alpha = torch.nn.Linear(model_params["HIDDEN_SIZE"], 1)

    def forward(self, x):
        """
        Args:
            x (Tensor): the sequence to the encoder (required).
                        shape (batch_size * num_assets, window_size, num_features_original)
        """
        _, n, d, device = x.shape[0], x.shape[1], x.shape[2], x.device  # n: window size; d: number of features

        stock_rep, _ = self.lstm(x)  # [batch_size * num_assets, window_size, num_features_original] -> [batch_size * num_assets, window_size, HIDDEN_SIZE]
        stock_rep = self.dropout(stock_rep)
        alpha_rep_1 = self.linear_1(stock_rep) # [batch_size * num_assets, window_size, HIDDEN_SIZE]
        alpha_rep_2 = self.linear_2(stock_rep[:, -1, :]).unsqueeze(1)  # [batch_size * num_assets, 1, HIDDEN_SIZE]
        # alpha = torch.sum(self.w_alpha * F.tanh(alpha_rep_1 + alpha_rep_2), dim=-1) # [S, L]
        alpha = self.w_alpha(F.tanh(alpha_rep_1 + alpha_rep_2)).squeeze(-1)
        alpha = F.softmax(alpha, dim=-1).unsqueeze(-1)  # [S, L, 1]
        stock_rep = torch.sum(stock_rep * alpha, dim=1)  # [S, H]

        return stock_rep  # [batch_size * num_assets, HIDDEN_SIZE]


class CAAN(nn.Module):
    """
    This class implements the Cross Asset Attention Network (CAAN) module
    """
    def __init__(self, model_params):
        super().__init__()
        self.linear_query = torch.nn.Linear(model_params["HIDDEN_SIZE"], model_params["HIDDEN_SIZE"])
        self.linear_key = torch.nn.Linear(model_params["HIDDEN_SIZE"], model_params["HIDDEN_SIZE"])
        self.linear_value = torch.nn.Linear(model_params["HIDDEN_SIZE"], model_params["HIDDEN_SIZE"])
        self.linear_winner = torch.nn.Linear(model_params["HIDDEN_SIZE"], 1)

    def forward(self, x):
        query = self.linear_query(x)  # [batch_size, num_assets, HIDDEN_SIZE]
        key = self.linear_key(x)  # [batch_size, num_assets, HIDDEN_SIZE]
        value = self.linear_value(x)  # [batch_size, num_assets, HIDDEN_SIZE]

        beta = torch.matmul(query, key.transpose(1, 2)) / torch.sqrt(torch.tensor(float(query.shape[-1])))  # [batch_size, num_assets, HIDDEN_SIZE]
        beta = F.softmax(beta, dim=-1).unsqueeze(-1)
        x = torch.sum(value.unsqueeze(1) * beta, dim=2)  # [batch_size, num_assets, HIDDEN_SIZE]

        final_scores = self.linear_winner(x).squeeze(-1)  # [batch_size, num_assets]

        return final_scores


class AlphaStock(nn.Module):
    """
    Class to generate predicted scores for the input assets based on the AlphaStock model.

    The AlphaStock model is a LSTM-based model for asset scoring and portfolio selection. It consists of two
    main components:

    1. Long Short-Term Memory with History state Attention (LSTM-HA): This module takes the input features for each asset
    over a time window and generates a fixed-size vector to represent the asset.
    A major advantage of LSTM-HA is that it can learn both the sequential and global dependences from stock history states.
    Compared with the existing studies that only use a recurrent neural network to extract the sequential dependence in history states

    2. Cross Asset Attention Network (CAAN): This module takes the sequence representations generated by the LSTM-HA and
    applies cross-asset attention to produce the final asset scores.

    The AlphaStock model takes an input tensor ``x`` of shape ``(batch_size, num_assets, num_features_augmented)``,
    where ``num_features_augmented`` represents the number of features (including any preprocessed or augmented
    features) for each asset. The final output of the AlphaStock model is a tensor of shape ``(batch_size, num_assets)``,
    where each element represents the predicted score for the corresponding asset.

    For more details, please refer to the paper `AlphaStock: A Buying-Winners-and-Selling-Losers Investment
    Strategy using Interpretable Deep Reinforcement Attention Networks <https://dl.acm.org/doi/abs/10.1145/3292500.3330647>`__.

    :param model_args: Dictionary containing model arguments, such as the number of features.
    :param model_params: Dictionary containing model hyperparameters, such as the number of layers, the hidden size, and the dropout rate.

    Example:
        .. code:: python
        >>> from finol.data_layer.dataset_loader import DatasetLoader
        >>> from finol.model_layer.model_instantiator import ModelInstantiator
        >>> from finol.utils import load_config, update_config, portfolio_selection
        >>>
        >>> # Configuration
        >>> config = load_config()
        >>> config["MODEL_NAME"] = "AlphaStock"
        >>> config["MODEL_PARAMS"]["AlphaStock"]["NUM_LAYERS"] = 1
        >>> config["MODEL_PARAMS"]["AlphaStock"]["HIDDEN_SIZE"] = 64
        >>> config["MODEL_PARAMS"]["AlphaStock"]["DROPOUT"] = 0.1
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

    \\
    """
    def __init__(self, model_args, model_params):
        super().__init__()
        self.config = load_config()
        self.model_args = model_args
        self.model_params = model_params

        self.lstm_ha = LSTM_HA(model_args, model_params)
        self.caan = CAAN(model_params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        :param x: Input tensor of shape ``(batch_size, num_assets, num_features_augmented)``.
        :return: Output tensor of shape ``(batch_size, num_assets)`` containing the predicted scores for each asset.
        """
        batch_size, num_assets, num_features_augmented = x.shape

        """Input Transformation"""
        x = x.view(batch_size, num_assets, self.model_args["window_size"], self.model_args["num_features_original"])
        x = rearrange(x, "b m n d -> (b m) n d")
        if self.config["SCALER"].startswith("Window"):
            x = ScalerSelector().window_normalize(x)

        """Sequence Representations Extraction (SREM)"""
        stock_rep = self.lstm_ha(x)  # [batch_size * num_assets, window_size, HIDDEN_SIZE] -> [batch_size * num_assets, HIDDEN_SIZE]
        x = stock_rep.view(batch_size, num_assets, self.model_params["HIDDEN_SIZE"])  # [batch_size * num_assets, HIDDEN_SIZE] -> [batch_size, num_assets, HIDDEN_SIZE]

        """Cross Asset Attention Network (CAAN)"""
        final_scores = self.caan(x)

        return final_scores