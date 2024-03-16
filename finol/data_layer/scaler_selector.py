from sklearn import preprocessing
from rich import print
from finol.config import *

scaler_dict = {
    'MinMaxScaler': preprocessing.MinMaxScaler,
    'MaxAbsScaler': preprocessing.MaxAbsScaler,
    'RobustScaler': preprocessing.RobustScaler,
    'StandardScaler': preprocessing.StandardScaler,
}


def select_scaler():
    scaler_cls = scaler_dict.get(SCALER, None)
    if scaler_cls is None:
        raise ValueError(f"Invalid scaler name: {SCALER}. Supported scaler are: {scaler_dict.keys()}")

    scaler = scaler_cls()
    return scaler