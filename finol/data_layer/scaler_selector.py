from sklearn import preprocessing
from finol.config import *

scaler_dict = {
    'None': 'original data',
    'StandardScaler': preprocessing.StandardScaler,
    'MinMaxScaler': preprocessing.MinMaxScaler,
    'MaxAbsScaler': preprocessing.MaxAbsScaler,
    'RobustScaler': preprocessing.RobustScaler,
}


def select_scaler():
    scaler_cls = scaler_dict.get(SCALER, None)
    if scaler_cls is None:
        raise ValueError(f"Invalid scaler name: {SCALER}. Supported scaler are: {scaler_dict.keys()}")

    if scaler_cls == 'original data':
        return None

    scaler = scaler_cls()
    return scaler