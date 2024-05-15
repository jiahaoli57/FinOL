import time

import torch
from finol.config import *


def LOG_WEALTH(portfolios, labels):
    dot_product = torch.sum(portfolios * labels, dim=-1)
    log_max_dot_product = torch.log(torch.clamp(dot_product, min=1e-6))
    loss = - torch.mean(log_max_dot_product)
    return loss


# def MIN_LOG_WEALTH(preds, labels):
#     dot_product = torch.sum(preds * labels, dim=-1)
#     log_max_dot_product = torch.log(torch.clamp(dot_product, min=1e-6))
#     loss = torch.mean(log_max_dot_product)
#     return loss


def LOG_WEALTH_L2_DIVERSIFICATION(portfolios, labels):
    dot_product = torch.sum(portfolios * labels, dim=-1)
    log_max_dot_product = torch.log(torch.clamp(dot_product, min=1e-6))

    # Calculate the similarity loss
    # We use the L2 norm to measure the difference between consecutive predictions
    diff = portfolios[1:, :] - portfolios[:-1, :]
    diff_loss = torch.norm(diff, p=2, dim=-1)

    # By minimizing the L2 norm of the difference vectors, you will tend to get two vectors that are similar, since the
    # L2 norm measures the Euclidean distance between vectors. When the L2 norm is small, it means that the difference
    # between two vectors is small.
    loss = - torch.mean(log_max_dot_product) + LAMBDA_L2 * torch.mean(diff_loss)
    return loss


def LOG_WEALTH_L2_CONCENTRATION(portfolios, labels):
    dot_product = torch.sum(portfolios * labels, dim=-1)
    log_max_dot_product = torch.log(torch.clamp(dot_product, min=1e-6))

    # Calculate the similarity loss
    # We use the L2 norm to measure the difference between consecutive predictions
    diff = portfolios[1:, :] - portfolios[:-1, :]
    diff_loss = torch.norm(diff, p=2, dim=-1)

    # By minimizing the L2 norm of the difference vectors, you will tend to get two vectors that are similar, since the
    # L2 norm measures the Euclidean distance between vectors. When the L2 norm is small, it means that the difference
    # between two vectors is small.
    loss = - torch.mean(log_max_dot_product) - LAMBDA_L2 * torch.mean(diff_loss)
    return loss


# def LOG_SINGLE_PERIOD_WEALTH_L2_DIVERSIFICATION(preds, labels):
#     dot_product = torch.sum(preds * labels, dim=-1)
#     log_max_dot_product = torch.log(torch.clamp(dot_product, min=1e-6))
#     l2_norm = torch.norm(preds, p=2, dim=-1)
#     loss = - torch.mean(log_max_dot_product) + LAMBDA_L2 * torch.mean(l2_norm)
#     return loss
#
#
# def LOG_SINGLE_PERIOD_WEALTH_L2_CONCENTRATION(preds, labels):
#     dot_product = torch.sum(preds * labels, dim=-1)
#     log_max_dot_product = torch.log(torch.clamp(dot_product, min=1e-6))
#     l2_norm = torch.norm(preds, p=2, dim=-1)
#     loss = - torch.mean(log_max_dot_product) - LAMBDA_L2 * torch.mean(l2_norm)
#     return loss


# def L2_DIVERSIFICATION(preds, labels):
#     l2_norm = torch.norm(preds, p=2, dim=-1)
#     loss = torch.mean(l2_norm)
#     return loss
#
#
# def L2_CONCENTRATION(preds, labels):
#     l2_norm = torch.norm(preds, p=2, dim=-1)
#     loss = - torch.mean(l2_norm)
#     return loss


class select_criterion:
    def __init__(self):
        self.criterion_dict = {
            'LOG_WEALTH': LOG_WEALTH,
            'LOG_WEALTH_L2_DIVERSIFICATION': LOG_WEALTH_L2_DIVERSIFICATION,
            'LOG_WEALTH_L2_CONCENTRATION': LOG_WEALTH_L2_CONCENTRATION,
            # 'LOG_SINGLE_PERIOD_WEALTH_L2_DIVERSIFICATION': LOG_SINGLE_PERIOD_WEALTH_L2_DIVERSIFICATION,
            # 'LOG_SINGLE_PERIOD_WEALTH_L2_CONCENTRATION': LOG_SINGLE_PERIOD_WEALTH_L2_CONCENTRATION,
            # 'L2_DIVERSIFICATION': LOG_SINGLE_PERIOD_WEALTH_L2_CONCENTRATION,
            # 'L2_CONCENTRATION': LOG_SINGLE_PERIOD_WEALTH_L2_CONCENTRATION,
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
