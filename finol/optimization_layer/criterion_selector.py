import torch
from rich import print
from finol.config import *


def LOG_SINGLE_PERIOD_WEALTH(preds, labels):
    dot_product = torch.sum(preds * labels, dim=-1)
    # log_max_dot_product = torch.log(dot_product)
    log_max_dot_product = torch.log(torch.clamp(dot_product, min=1e-6))
    loss = - torch.mean(log_max_dot_product)
    return loss


def LOG_SINGLE_PERIOD_WEALTH_L2_DIVERSIFICATION(preds, labels):
    dot_product = torch.sum(preds * labels, dim=-1)
    log_max_dot_product = torch.log(torch.clamp(dot_product, min=1e-6))
    l2_norm = torch.norm(preds, p=2, dim=-1)
    loss = - torch.mean(log_max_dot_product) + LAMBDA_L2 * torch.mean(l2_norm)
    return loss


def LOG_SINGLE_PERIOD_WEALTH_L2_CONCENTRATION(preds, labels):
    dot_product = torch.sum(preds * labels, dim=-1)
    log_max_dot_product = torch.log(torch.clamp(dot_product, min=1e-6))
    l2_norm = torch.norm(preds, p=2, dim=-1)
    loss = - torch.mean(log_max_dot_product) - LAMBDA_L2 * torch.mean(l2_norm)
    return loss


def L2_DIVERSIFICATION(preds, labels):
    l2_norm = torch.norm(preds, p=2, dim=-1)
    loss = torch.mean(l2_norm)
    return loss


def L2_CONCENTRATION(preds, labels):
    l2_norm = torch.norm(preds, p=2, dim=-1)
    loss = - torch.mean(l2_norm)
    return loss


class select_criterion:
    def __init__(self):
        self.criterion_dict = {
            'LOG_SINGLE_PERIOD_WEALTH': LOG_SINGLE_PERIOD_WEALTH,
            'LOG_SINGLE_PERIOD_WEALTH_L2_DIVERSIFICATION': LOG_SINGLE_PERIOD_WEALTH_L2_DIVERSIFICATION,
            'LOG_SINGLE_PERIOD_WEALTH_L2_CONCENTRATION': LOG_SINGLE_PERIOD_WEALTH_L2_CONCENTRATION,
            'L2_DIVERSIFICATION': LOG_SINGLE_PERIOD_WEALTH_L2_CONCENTRATION,
            'L2_CONCENTRATION': LOG_SINGLE_PERIOD_WEALTH_L2_CONCENTRATION,
        }
        self.default_criterion = CRITERION_NAME

    def __call__(self, preds, labels):
        criterion = self.criterion_dict.get(
            self.default_criterion,
            None
        )

        if criterion is None:
            raise ValueError(f"Invalid criterion: {self.default_criterion}")

        return criterion(preds, labels)
