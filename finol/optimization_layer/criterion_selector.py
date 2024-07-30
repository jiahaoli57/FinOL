import torch

from finol.utils import load_config


class CriterionSelector:
    def __init__(self):
        self.config = load_config()

    def LogWealth(self, portfolios, labels):
        daily_returns = torch.sum(portfolios * labels, dim=-1)
        log_returns = torch.log(torch.clamp(daily_returns, min=1e-6))
        loss = - torch.mean(log_returns)
        return loss

    def LogWealth_L2Diversification(self, portfolios, labels):
        daily_returns = torch.sum(portfolios * labels, dim=-1)
        log_returns = torch.log(torch.clamp(daily_returns, min=1e-6))

        # Calculate the similarity loss
        # We use the L2 norm to measure the difference between consecutive predictions
        diff = portfolios[1:, :] - portfolios[:-1, :]
        diff_loss = torch.norm(diff, p=2, dim=-1)

        # By minimizing the L2 norm of the difference vectors, you will tend to get two vectors that are similar, since the
        # L2 norm measures the Euclidean distance between vectors. When the L2 norm is small, it means that the difference
        # between two vectors is small.
        loss = - torch.mean(log_returns) + self.config["LAMBDA_L2"] * torch.mean(diff_loss)
        return loss

    def LogWealth_L2Concentration(self, portfolios, labels):
        daily_returns = torch.sum(portfolios * labels, dim=-1)
        log_returns = torch.log(torch.clamp(daily_returns, min=1e-6))

        # Calculate the similarity loss
        # We use the L2 norm to measure the difference between consecutive predictions
        diff = portfolios[1:, :] - portfolios[:-1, :]
        diff_loss = torch.norm(diff, p=2, dim=-1)

        # By minimizing the L2 norm of the difference vectors, you will tend to get two vectors that are similar, since the
        # L2 norm measures the Euclidean distance between vectors. When the L2 norm is small, it means that the difference
        # between two vectors is small.
        loss = - torch.mean(log_returns) - self.config["LAMBDA_L2"] * torch.mean(diff_loss)
        return loss

    def L2Diversification(self, portfolios, labels):
        # Calculate the similarity loss
        # We use the L2 norm to measure the difference between consecutive predictions
        diff = portfolios[1:, :] - portfolios[:-1, :]
        diff_loss = torch.norm(diff, p=2, dim=-1)

        # By minimizing the L2 norm of the difference vectors, you will tend to get two vectors that are similar, since the
        # L2 norm measures the Euclidean distance between vectors. When the L2 norm is small, it means that the difference
        # between two vectors is small.
        loss = torch.mean(diff_loss)
        return loss

    def L2Concentration(self, portfolios, labels):
        # Calculate the similarity loss
        # We use the L2 norm to measure the difference between consecutive predictions
        diff = portfolios[1:, :] - portfolios[:-1, :]
        diff_loss = torch.norm(diff, p=2, dim=-1)

        # By minimizing the L2 norm of the difference vectors, you will tend to get two vectors that are similar, since the
        # L2 norm measures the Euclidean distance between vectors. When the L2 norm is small, it means that the difference
        # between two vectors is small.
        loss = - torch.mean(diff_loss)
        return loss

    def SharpeRatio(self, portfolios, labels):
        daily_returns = torch.sum(portfolios * labels, dim=-1)
        # log_returns = torch.log(torch.clamp(daily_returns, min=1e-6))
        excess_returns = daily_returns  # risk_free_rate = 0

        mean_excess_return = torch.mean(excess_returns)
        std_excess_return = torch.std(excess_returns)

        sharpe_ratio = mean_excess_return / (std_excess_return + 1e-6)
        loss = - sharpe_ratio
        return loss

    def Volatility(self, portfolios, labels):
        daily_returns = torch.sum(portfolios * labels, dim=-1)
        volatility = torch.std(daily_returns)
        loss = volatility
        return loss

    def __call__(self, preds, labels):
        criterion = getattr(self, self.config["CRITERION_NAME"])

        return criterion(preds, labels)
