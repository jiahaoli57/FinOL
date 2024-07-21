import torch

from sklearn import preprocessing
from finol.utils import load_config


def WindowStandardScaler(data):
    """
    Normalize the input tensor data along the window dimension via StandardScaler.

    Input shape:
        data: (..., window_size, feature_dim)

    Output shape:
        standardized_data: (..., window_size, feature_dim)
    """
    window_mean = torch.mean(data, dim=-2, keepdim=True)
    window_std = torch.std(data, dim=-2, keepdim=True) + 1e-7
    standardized_data = (data - window_mean) / window_std
    return standardized_data


def WindowMinMaxScaler(data):
    """
    Normalize the input tensor data along the window dimension via MinMaxScaler.

    Input shape:
        data: (..., window_size, feature_dim)

    Output shape:
        normalized_data: (..., window_size, feature_dim)
    """
    min_vals = torch.min(data, dim=-2, keepdim=True).values
    max_vals = torch.max(data, dim=-2, keepdim=True).values
    normalized_data = (data - min_vals) / (max_vals - min_vals + 1e-7)
    return normalized_data


def WindowMaxAbsScaler(data):
    """
    Normalize the input tensor data along the window dimension via MaxAbsScaler.

    Input shape:
        data: (..., window_size, feature_dim)

    Output shape:
        normalized_data: (..., window_size, feature_dim)
    """
    max_abs_vals = torch.max(torch.abs(data), dim=-2, keepdim=True).values
    normalized_data = data / (max_abs_vals + 1e-7)
    return normalized_data


def WindowRobustScaler(data):
    """
    Normalize the input tensor data along the window dimension via RobustScaler.

    Input shape:
        data: (..., window_size, feature_dim)

    Output shape:
        scaled_data: (..., window_size, feature_dim)
    """

    # Calculate the median and IQR (interquartile range) along the window dimension
    window_median = torch.median(data, dim=-2, keepdim=True).values
    q1 = torch.quantile(data, 0.25, dim=-2, keepdim=True)
    q3 = torch.quantile(data, 0.75, dim=-2, keepdim=True)
    window_iqr = q3 - q1 + 1e-7

    # Scale the data using the median and IQR
    normalized_data = (data - window_median) / window_iqr
    return normalized_data


scaler_dict = {
    "None": "None",
    "StandardScaler": preprocessing.StandardScaler,
    "MinMaxScaler": preprocessing.MinMaxScaler,
    "MaxAbsScaler": preprocessing.MaxAbsScaler,
    "RobustScaler": preprocessing.RobustScaler,
    "WindowStandardScaler": WindowStandardScaler,
    "WindowMinMaxScaler": WindowMinMaxScaler,
    "WindowMaxAbsScaler": WindowMaxAbsScaler,
    "WindowRobustScaler": WindowRobustScaler,
}


class ScalerSelector():
    def __init__(self):
        self.config = load_config()

    def select_scaler(self):
        scaler_cls = scaler_dict.get(self.config["SCALER"], None)
        if scaler_cls is None:
            raise ValueError(f"Invalid scaler name: {self.config['SCALER']}. Supported scalers are: {scaler_dict.keys()}")

        if self.config["SCALER"] == "None" or self.config["SCALER"].startswith("Window"):
            return None

        scaler = scaler_cls()
        return scaler

    def window_normalize(self, data):
        """
        Normalize the input tensor data along the window dimension based on different methods.

        Input shape:
            data: (..., window_size, feature_dim)

        Output shape:
            scaled_data: (..., window_size, feature_dim)
        """
        scaler_cls = scaler_dict.get(self.config["SCALER"], None)
        if scaler_cls is None:
            raise ValueError(f"Invalid scaler name: {self.config['SCALER']}. Supported scalers are: {scaler_dict.keys()}")

        normalized_data = scaler_cls(data)
        return normalized_data


