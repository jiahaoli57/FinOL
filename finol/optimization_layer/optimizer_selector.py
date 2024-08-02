import torch
import torch_optimizer as optim

from typing import List, Tuple, Dict, Union, Any
from finol.utils import load_config


class OptimizerSelector:
    """
    Class to select optimizer for portfolio selection.
    """
    def __init__(self, model) -> None:
        self.config = load_config()
        self.model = model

        self.optimizer_dict = {
            "Adadelta": torch.optim.Adadelta,
            "Adagrad": torch.optim.Adagrad,
            "Adam": torch.optim.Adam,
            "AdamW": torch.optim.AdamW,
            # "SparseAdam": torch.optim.SparseAdam,
            "Adamax": torch.optim.Adamax,
            "ASGD": torch.optim.ASGD,
            "SGD": torch.optim.SGD,
            "RAdam": torch.optim.RAdam,
            "Rprop": torch.optim.Rprop,
            "RMSprop": torch.optim.RMSprop,
            "NAdam": torch.optim.NAdam,
            # "LBFGS": torch.optim.LBFGS,
            "A2GradExp": optim.A2GradExp,
            "A2GradInc": optim.A2GradInc,
            "A2GradUni": optim.A2GradUni,
            "AccSGD": optim.AccSGD,
            "AdaBelief": optim.AdaBelief,
            "AdaBound": optim.AdaBound,
            "AdaMod": optim.AdaMod,
            "Adafactor": optim.Adafactor,
            # "Adahessian": optim.Adahessian,
            "AdamP": optim.AdamP,
            "AggMo": optim.AggMo,
            "Apollo": optim.Apollo,
            "DiffGrad": optim.DiffGrad,
            "LARS": optim.LARS,
            "Lamb": optim.Lamb,
            "MADGRAD": optim.MADGRAD,
            "NovoGrad": optim.NovoGrad,
            "PID": optim.PID,
            "QHAdam": optim.QHAdam,
            "QHM": optim.QHM,
            "Ranger": optim.Ranger,
            "RangerQH": optim.RangerQH,
            "RangerVA": optim.RangerVA,
            "SGDP": optim.SGDP,
            "SGDW": optim.SGDW,
            "SWATS": optim.SWATS,
            # "Shampoo": optim.Shampoo,
            "Yogi": optim.Yogi
        }

    def select_optimizer(self, sampled_lr: Union[Dict[str, Any], None] = None, sampled_optimizer: Union[Dict[str, Any], None] = None) -> object:
        """
        Selects an optimizer based on the configuration.

        :param sampled_lr: Sampled learning rate or None. Required only when optimizing the learning rate selection.
        :param sampled_optimizer: Sampled optimizer or None. Required only when optimizing the optimizer selection.
        :return: The selected optimizer for the model.
        :raises ValueError: If the optimizer name is invalid.
        """
        if sampled_lr!= None:
            lr = sampled_lr
        else:
            lr = self.config["LEARNING_RATE"]
        if sampled_optimizer!= None:
            optimizer_name = sampled_optimizer
        else:
            optimizer_name = self.config["OPTIMIZER_NAME"]

        optimizer_cls = self.optimizer_dict.get(optimizer_name, None)
        if optimizer_cls is None:
            raise ValueError(f"Invalid optimizer name: {optimizer_name}. Supported optimizers are: {self.optimizer_dict}")
        return optimizer_cls(self.model.parameters(), lr=lr)