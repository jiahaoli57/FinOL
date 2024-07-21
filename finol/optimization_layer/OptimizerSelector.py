import torch
import torch_optimizer as optim

from finol.utils import *

optimizer_dict = {
    'Adadelta': torch.optim.Adadelta,
    'Adagrad': torch.optim.Adagrad,
    'Adam': torch.optim.Adam,
    'AdamW': torch.optim.AdamW,
    # 'SparseAdam': torch.optim.SparseAdam,
    'Adamax': torch.optim.Adamax,
    'ASGD': torch.optim.ASGD,
    'SGD': torch.optim.SGD,
    'RAdam': torch.optim.RAdam,
    'Rprop': torch.optim.Rprop,
    'RMSprop': torch.optim.RMSprop,
    'NAdam': torch.optim.NAdam,
    # 'LBFGS': torch.optim.LBFGS,
    'A2GradExp': optim.A2GradExp,
    'A2GradInc': optim.A2GradInc,
    'A2GradUni': optim.A2GradUni,
    'AccSGD': optim.AccSGD,
    'AdaBelief': optim.AdaBelief,
    'AdaBound': optim.AdaBound,
    'AdaMod': optim.AdaMod,
    'Adafactor': optim.Adafactor,
    # 'Adahessian': optim.Adahessian,
    'AdamP': optim.AdamP,
    'AggMo': optim.AggMo,
    'Apollo': optim.Apollo,
    'DiffGrad': optim.DiffGrad,
    'LARS': optim.LARS,
    'Lamb': optim.Lamb,
    'MADGRAD': optim.MADGRAD,
    'NovoGrad': optim.NovoGrad,
    'PID': optim.PID,
    'QHAdam': optim.QHAdam,
    'QHM': optim.QHM,
    'Ranger': optim.Ranger,
    'RangerQH': optim.RangerQH,
    'RangerVA': optim.RangerVA,
    'SGDP': optim.SGDP,
    'SGDW': optim.SGDW,
    'SWATS': optim.SWATS,
    # 'Shampoo': optim.Shampoo,
    'Yogi': optim.Yogi
}


class OptimizerSelector:
    def __init__(self, model):
        config = load_config()
        # Optimization Layer Configuration
        self.LEARNING_RATE = config["LEARNING_RATE"]
        self.OPTIMIZER_NAME = config["OPTIMIZER_NAME"]

        self.model = model

    def select_optimizer(self,  sampled_lr=None, sampled_optimizer=None):
        if sampled_lr!= None:
            self.LEARNING_RATE = sampled_lr
        if sampled_optimizer!= None:
            self.OPTIMIZER_NAME = sampled_optimizer

        optimizer_cls = optimizer_dict.get(self.OPTIMIZER_NAME, None)
        if optimizer_cls is None:
            raise ValueError(f"Invalid optimizer name: {self.OPTIMIZER_NAME}. Supported optimizers are: {optimizer_dict}")

        optimizer = optimizer_cls(self.model.parameters(), lr=self.LEARNING_RATE)
        return optimizer