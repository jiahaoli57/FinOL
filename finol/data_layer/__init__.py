from rich import print
from finol.data_layer.dataset_loader import DatasetLoader
from finol.data_layer.scaler_selector import ScalerSelector
from finol.utils import load_config, update_config, detect_device

__all__ = [
    "DatasetLoader",
    "ScalerSelector",
]

config = load_config()
detect_device(config)
update_config(config)