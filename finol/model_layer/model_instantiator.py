import torch.nn as nn

from typing import List, Tuple, Dict, Union
from finol.utils import load_config, set_seed
from finol.model_layer.AlphaPortfolio import AlphaPortfolio
from finol.model_layer.CNN import CNN
from finol.model_layer.DNN import DNN
from finol.model_layer.LSRE_CAAN import LSRE_CAAN
from finol.model_layer.LSTM import LSTM
from finol.model_layer.RNN import RNN
from finol.model_layer.Transformer import Transformer
from finol.model_layer.CustomModel import CustomModel

model_dict = {
    "AlphaPortfolio": AlphaPortfolio,
    "CNN": CNN,
    "DNN": DNN,
    "LSRE-CAAN": LSRE_CAAN,
    "LSTM": LSTM,
    "RNN": RNN,
    "Transformer": Transformer,
    "CustomModel": CustomModel,
}


class ModelInstantiator:
    """
    Class to instantiate model class based on the provided configuration.

    :param load_dataset_output: Dictionary containing output from function :func:`~finol.data_layer.DatasetLoader.load_dataset`.
    """
    def __init__(self, load_dataset_output: Dict):
        self.config = load_config()
        self.load_dataset_output = load_dataset_output
        set_seed(seed=self.config["MANUAL_SEED"])

        if "AlphaPortfolio" in self.config["MODEL_PARAMS"] and self.config["MODEL_NAME"] == "AlphaPortfolio":
            self.model_args = {
                "num_features_original": self.load_dataset_output["num_features_original"],
                "window_size": self.load_dataset_output["window_size"],
            }
            self.model_params = {
                "NUM_LAYERS": self.config["MODEL_PARAMS"]["AlphaPortfolio"]["NUM_LAYERS"],
                "DIM_EMBEDDING": self.config["MODEL_PARAMS"]["AlphaPortfolio"]["DIM_EMBEDDING"],
                "DIM_FEEDFORWARD": self.config["MODEL_PARAMS"]["AlphaPortfolio"]["DIM_FEEDFORWARD"],
                "NUM_HEADS": self.config["MODEL_PARAMS"]["AlphaPortfolio"]["NUM_HEADS"],
                "DROPOUT": self.config["MODEL_PARAMS"]["AlphaPortfolio"]["DROPOUT"],
            }

        if "CNN" in self.config["MODEL_PARAMS"] and self.config["MODEL_NAME"] == "CNN":
            self.model_args = {
                "num_features_original": self.load_dataset_output["num_features_original"],
                "window_size": self.load_dataset_output["window_size"],
            }
            self.model_params = {
                "OUT_CHANNELS": self.config["MODEL_PARAMS"]["CNN"]["OUT_CHANNELS"],
                "KERNEL_SIZE": self.config["MODEL_PARAMS"]["CNN"]["KERNEL_SIZE"],
                "STRIDE": self.config["MODEL_PARAMS"]["CNN"]["STRIDE"],
                "HIDDEN_SIZE": self.config["MODEL_PARAMS"]["CNN"]["HIDDEN_SIZE"],
                "DROPOUT": self.config["MODEL_PARAMS"]["CNN"]["DROPOUT"],
            }

        if "DNN" in self.config["MODEL_PARAMS"] and self.config["MODEL_NAME"] == "DNN":
            self.model_args = {
                "num_features_augmented": self.load_dataset_output["num_features_augmented"],
            }
            self.model_params = {
                "NUM_LAYERS": self.config["MODEL_PARAMS"]["DNN"]["NUM_LAYERS"],
                "HIDDEN_SIZE": self.config["MODEL_PARAMS"]["DNN"]["HIDDEN_SIZE"],
                "DROPOUT": self.config["MODEL_PARAMS"]["DNN"]["DROPOUT"],
            }

        if "LSRE-CAAN" in self.config["MODEL_PARAMS"] and self.config["MODEL_NAME"].startswith("LSRE-CAAN"):
            self.model_args = {
                "num_features_original": self.load_dataset_output["num_features_original"],
                "window_size": self.load_dataset_output["window_size"],
            }
            self.model_params = {
                "NUM_LAYERS": self.config["MODEL_PARAMS"]["LSRE-CAAN"]["NUM_LAYERS"],
                "NUM_LATENTS": self.config["MODEL_PARAMS"]["LSRE-CAAN"]["NUM_LATENTS"],
                "LATENT_DIM": self.config["MODEL_PARAMS"]["LSRE-CAAN"]["LATENT_DIM"],
                "CROSS_HEADS": self.config["MODEL_PARAMS"]["LSRE-CAAN"]["CROSS_HEADS"],
                "LATENT_HEADS": self.config["MODEL_PARAMS"]["LSRE-CAAN"]["LATENT_HEADS"],
                "CROSS_DIM_HEAD": self.config["MODEL_PARAMS"]["LSRE-CAAN"]["CROSS_DIM_HEAD"],
                "LATENT_DIM_HEAD": self.config["MODEL_PARAMS"]["LSRE-CAAN"]["LATENT_DIM_HEAD"],
                "DROPOUT": self.config["MODEL_PARAMS"]["LSRE-CAAN"]["DROPOUT"],
            }

        if "LSTM" in self.config["MODEL_PARAMS"] and self.config["MODEL_NAME"] == "LSTM":
            self.model_args = {
                "num_features_original": self.load_dataset_output["num_features_original"],
                "window_size": self.load_dataset_output["window_size"],
            }
            self.model_params = {
                "NUM_LAYERS": self.config["MODEL_PARAMS"]["LSTM"]["NUM_LAYERS"],
                "HIDDEN_SIZE": self.config["MODEL_PARAMS"]["LSTM"]["HIDDEN_SIZE"],
                "DROPOUT": self.config["MODEL_PARAMS"]["LSTM"]["DROPOUT"],
            }

        if "RNN" in self.config["MODEL_PARAMS"] and self.config["MODEL_NAME"] == "RNN":
            self.model_args = {
                "num_features_original": self.load_dataset_output["num_features_original"],
                "window_size": self.load_dataset_output["window_size"],
            }
            self.model_params = {
                "NUM_LAYERS": self.config["MODEL_PARAMS"]["RNN"]["NUM_LAYERS"],
                "HIDDEN_SIZE": self.config["MODEL_PARAMS"]["RNN"]["HIDDEN_SIZE"],
                "DROPOUT": self.config["MODEL_PARAMS"]["RNN"]["DROPOUT"],
            }

        if "Transformer" in self.config["MODEL_PARAMS"] and self.config["MODEL_NAME"] == "Transformer":
            self.model_args = {
                "num_features_original": self.load_dataset_output["num_features_original"],
                "window_size": self.load_dataset_output["window_size"],
            }
            self.model_params = {
                "NUM_LAYERS": self.config["MODEL_PARAMS"]["Transformer"]["NUM_LAYERS"],
                "DIM_EMBEDDING": self.config["MODEL_PARAMS"]["Transformer"]["DIM_EMBEDDING"],
                "DIM_FEEDFORWARD": self.config["MODEL_PARAMS"]["Transformer"]["DIM_FEEDFORWARD"],
                "NUM_HEADS": self.config["MODEL_PARAMS"]["Transformer"]["NUM_HEADS"],
                "DROPOUT": self.config["MODEL_PARAMS"]["Transformer"]["DROPOUT"],
            }

        if "CustomModel" in self.config["MODEL_PARAMS"] and self.config["MODEL_NAME"] == "CustomModel":
            self.model_args = {}
            self.model_params = {}

    def instantiate_model(self, sampled_params=None) -> nn.Module:
        """
        Instantiate a model based on the configuration provided in the class.

        This method retrieves the model class defined in the configuration, and creates an
        instance of the model that is moved to the designated device.

        :return: An instance of the specified model class.
        """
        model_cls = model_dict.get(self.config["MODEL_NAME"], None)
        if model_cls is None:
            raise ValueError(f"Invalid model: {self.config['MODEL_NAME']}. Supported models are: {model_dict}")

        if self.config["TUNE_PARAMETERS"] and sampled_params != None:
            self.model_params = sampled_params

        model = model_cls(self.model_args, self.model_params).to(self.config["DEVICE"])
        return model
