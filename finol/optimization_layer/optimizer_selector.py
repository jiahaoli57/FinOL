import torch
import torch_optimizer as optim

from finol.config import *

optimizer_dict = {
    'Adadelta': torch.optim.Adadelta,
    'Adagrad': torch.optim.Adagrad,
    'Adam': torch.optim.Adam,
    'AdamW': torch.optim.AdamW,
    'SparseAdam': torch.optim.SparseAdam,
    'Adamax': torch.optim.Adamax,
    'ASGD': torch.optim.ASGD,
    'SGD': torch.optim.SGD,
    'RAdam': torch.optim.RAdam,
    'Rprop': torch.optim.Rprop,
    'RMSprop': torch.optim.RMSprop,
    'NAdam': torch.optim.NAdam,
    'LBFGS': torch.optim.LBFGS,
    'A2GradExp': optim.A2GradExp,
    'A2GradInc': optim.A2GradInc,
    'A2GradUni': optim.A2GradUni,
    'AccSGD': optim.AccSGD,
    'AdaBelief': optim.AdaBelief,
    'AdaBound': optim.AdaBound,
    'AdaMod': optim.AdaMod,
    'Adafactor': optim.Adafactor,
    'Adahessian': optim.Adahessian,
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
    'Shampoo': optim.Shampoo,
    'Yogi': optim.Yogi
}


def select_optimizer(model):
    optimizer_cls = optimizer_dict.get(OPTIMIZER_NAME, None)
    if optimizer_cls is None:
        raise ValueError(f"Invalid optimizer name: {OPTIMIZER_NAME}. Supported optimizers are: {optimizer_dict}")

    optimizer = optimizer_cls(model.parameters(), lr=LEARNING_RATE)
    return optimizer