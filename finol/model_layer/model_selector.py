import torch
import torch_optimizer as optim
from rich import print
from finol.model_layer.DNN import *
from finol.model_layer.RNN import *
from finol.model_layer.LSTM import *
from finol.model_layer.CNN import *
from finol.model_layer.Transformer import *
from finol.model_layer.LSRE_CAAN import *
from finol.config import *

model_dict = {
    'DNN': DNN,
    'RNN': RNN,
    'LSTM': LSTM,
    'CNN': CNN,
    'Transformer': Transformer,
    'LSRE-CAAN': LSRE_CAAN
}


def select_model(load_dataset_output):
    model_cls = model_dict.get(MODEL_NAME, None)
    if model_cls is None:
        raise ValueError(f"Invalid model strategy: {MODEL_NAME}")

    model = model_cls(
        num_assets=load_dataset_output['NUM_ASSETS'],
        num_features_augmented=load_dataset_output['NUM_FEATURES_AUGMENTED'],
        num_features_original=load_dataset_output['NUM_FEATURES_ORIGINAL'],
        window_size=load_dataset_output['WINDOW_SIZE']
    ).to(DEVICE)
    return model

