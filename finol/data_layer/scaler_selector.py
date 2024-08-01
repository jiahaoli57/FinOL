import torch

from typing import List, Tuple, Dict, Union
from sklearn import preprocessing
from finol.utils import load_config


class ScalerSelector:
    """
    Class to select different types of scalers and performs partial data standardization depending on the configuration.
    """
    def __init__(self) -> None:
        self.config = load_config()
        self.scaler_dict = {
            "None": "None",
            "StandardScaler": preprocessing.StandardScaler,
            "MinMaxScaler": preprocessing.MinMaxScaler,
            "MaxAbsScaler": preprocessing.MaxAbsScaler,
            "RobustScaler": preprocessing.RobustScaler,
            "WindowStandardScaler": self.window_normalize_via_StandardScaler,
            "WindowMinMaxScaler": self.window_normalize_via_MinMaxScaler,
            "WindowMaxAbsScaler": self.window_normalize_via_MaxAbsScaler,
            "WindowRobustScaler": self.window_normalize_via_RobustScaler,
        }

    def select_scaler(self) -> Union[object, None]:
        """
        Select the data scaler based on the configuration.

        :return: The selected scaler object, or None if no scaler is specified.
        """
        scaler_cls = self.scaler_dict.get(self.config["SCALER"], None)
        if scaler_cls is None:
            raise ValueError(f"Invalid scaler name: {self.config['SCALER']}. Supported scalers are: {self.scaler_dict.keys()}")

        if self.config["SCALER"] == "None" or self.config["SCALER"].startswith("Window"):
            return None

        scaler = scaler_cls()
        return scaler

    def window_normalize_via_StandardScaler(self, data: torch.Tensor) -> torch.Tensor:
        """
        Normalize the input tensor data along the window dimension via StandardScaler.

        :param data: Input data tensor of shape (..., window_size, feature_dim).
        :return: Standardized data tensor of the same shape.
        """
        window_mean = torch.mean(data, dim=-2, keepdim=True)
        window_std = torch.std(data, dim=-2, keepdim=True) + 1e-7
        standardized_data = (data - window_mean) / window_std
        return standardized_data

    def window_normalize_via_MinMaxScaler(self, data: torch.Tensor) -> torch.Tensor:
        """
        Normalize the input tensor data along the window dimension via MinMaxScaler.

        :param data: Input data tensor of shape (..., window_size, feature_dim).
        :return: Standardized data tensor of the same shape.
        """
        min_vals = torch.min(data, dim=-2, keepdim=True).values
        max_vals = torch.max(data, dim=-2, keepdim=True).values
        normalized_data = (data - min_vals) / (max_vals - min_vals + 1e-7)
        return normalized_data

    def window_normalize_via_MaxAbsScaler(self, data: torch.Tensor) -> torch.Tensor:
        """
        Normalize the input tensor data along the window dimension via MaxAbsScaler.

        :param data: Input data tensor of shape (..., window_size, feature_dim).
        :return: Standardized data tensor of the same shape.
        """
        max_abs_vals = torch.max(torch.abs(data), dim=-2, keepdim=True).values
        normalized_data = data / (max_abs_vals + 1e-7)
        return normalized_data

    def window_normalize_via_RobustScaler(self, data: torch.Tensor) -> torch.Tensor:
        """
        Normalize the input tensor data along the window dimension via RobustScaler.

        :param data: Input data tensor of shape (..., window_size, feature_dim).
        :return: Standardized data tensor of the same shape.
        """

        # Calculate the median and IQR (interquartile range) along the window dimension
        window_median = torch.median(data, dim=-2, keepdim=True).values
        q1 = torch.quantile(data, 0.25, dim=-2, keepdim=True)
        q3 = torch.quantile(data, 0.75, dim=-2, keepdim=True)
        window_iqr = q3 - q1 + 1e-7

        # Scale the data using the median and IQR
        normalized_data = (data - window_median) / window_iqr
        return normalized_data

    def window_normalize(self, data: torch.Tensor) -> torch.Tensor:
        """
        Normalize the input tensor data along the window dimension based on different methods.

        :param data: Input data tensor of shape (..., window_size, feature_dim).
        :return: Standardized data tensor of the same shape.
        """
        scaler_cls = self.scaler_dict.get(self.config["SCALER"], None)
        if scaler_cls is None:
            raise ValueError(f"Invalid scaler name: {self.config['SCALER']}. Supported scalers are: {self.scaler_dict.keys()}")

        normalized_data = scaler_cls(data)
        return normalized_data


