import time
import torch

from finol.utils import load_config


class CriterionSelector:
    """
    Class to select and compute loss criterion for portfolio selection.
    """
    def __init__(self) -> None:
        self.config = load_config()
        self.criterion_dict = {
            "LogWealth": self.compute_log_wealth_loss,
            "LogWealthL2Diversification": self.compute_log_wealth_l2_diversification_loss,
            "LogWealthL2Concentration": self.compute_log_wealth_l2_concentration_loss,
            "L2Diversification": self.compute_l2_diversification_loss,
            "L2Concentration": self.compute_l2_concentration_loss,
            "SharpeRatio": self.compute_sharpe_ratio_loss,
            "Volatility": self.compute_volatility_loss,
            "CustomCriterion": self.compute_custom_criterion_loss,
        }

    def compute_log_wealth_loss(self, portfolios: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute the ``LogWealth`` loss, which calculates the negative mean logarithm of the portfolio's daily returns.

        :param portfolios: Portfolio weights tensor of shape (batch_size, num_assets).
        :param labels: Label tensor representing asset returns of shape (batch_size, num_assets).
        :return: ``LogWealth`` loss tensor, representing the negative mean logarithm of the portfolio's daily returns.
        """
        daily_returns = torch.sum(portfolios * labels, dim=-1)
        log_returns = torch.log(torch.clamp(daily_returns, min=1e-6))
        loss = - torch.mean(log_returns)
        return loss

    def compute_log_wealth_l2_diversification_loss(self, portfolios: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute the ``LogWealthL2Diversification`` loss, which extends the ``LogWealth`` loss by incorporating diversification measures.

        This loss function calculates the negative mean logarithm of the portfolio's daily returns while considering diversification metrics.

        :param portfolios: Portfolio weights tensor of shape (batch_size, num_assets).
        :param labels: Label tensor representing asset returns of shape (batch_size, num_assets).
        :return: ``LogWealthL2Diversification`` loss tensor, a modified version of ``LogWealth`` loss with diversification metrics included.
        """
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

    def compute_log_wealth_l2_concentration_loss(self, portfolios: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute the ``LogWealthL2Concentration`` loss, which extends the ``LogWealth`` loss by incorporating concentration measures.

        :param portfolios: Portfolio weights tensor of shape (batch_size, num_assets).
        :param labels: Label tensor representing asset returns of shape (batch_size, num_assets).
        :return: ``LogWealthL2Concentration`` loss tensor, a modified version of ``LogWealth`` loss with concentration metrics included.
        """
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

    def compute_l2_diversification_loss(self, portfolios: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute the ``L2Diversification`` loss, which measures the diversification of a portfolio based on the L2 norm of consecutive portfolio weight differences.

        This loss function calculates the mean L2 norm of the differences between consecutive portfolio weight vectors.

        :param portfolios: Portfolio weights tensor of shape (batch_size, num_assets).
        :param labels: Label tensor representing asset returns of shape (batch_size, num_assets).
        :return: ``L2Diversification`` loss tensor, indicating the degree of diversification based on the L2 norm of portfolio weight differences.
        """
        # Calculate the similarity loss
        # We use the L2 norm to measure the difference between consecutive predictions
        diff = portfolios[1:, :] - portfolios[:-1, :]
        diff_loss = torch.norm(diff, p=2, dim=-1)

        # By minimizing the L2 norm of the difference vectors, you will tend to get two vectors that are similar, since the
        # L2 norm measures the Euclidean distance between vectors. When the L2 norm is small, it means that the difference
        # between two vectors is small.
        loss = torch.mean(diff_loss)
        return loss

    def compute_l2_concentration_loss(self, portfolios: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute the ``L2Concentration`` loss, which measures the concentration of a portfolio based on the L2 norm of consecutive portfolio weight differences.

        This loss function calculates the negative mean L2 norm of the differences between consecutive portfolio weight vectors.

        :param portfolios: Portfolio weights tensor of shape (batch_size, num_assets).
        :param labels: Label tensor representing asset returns of shape (batch_size, num_assets).
        :return: ``L2Concentration`` loss tensor, indicating the degree of concentration based on the negative mean L2 norm of portfolio weight differences.
        """
        # Calculate the similarity loss
        # We use the L2 norm to measure the difference between consecutive predictions
        diff = portfolios[1:, :] - portfolios[:-1, :]
        diff_loss = torch.norm(diff, p=2, dim=-1)

        # By minimizing the L2 norm of the difference vectors, you will tend to get two vectors that are similar, since the
        # L2 norm measures the Euclidean distance between vectors. When the L2 norm is small, it means that the difference
        # between two vectors is small.
        loss = - torch.mean(diff_loss)
        return loss

    def compute_sharpe_ratio_loss(self, portfolios: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute the ``SharpeRatio`` loss, which evaluates the risk-adjusted return of the portfolios using the Sharpe Ratio.

        This loss function calculates the negative Sharpe Ratio, which is the ratio of the mean excess return to the standard deviation of the excess return.

        :param portfolios: Portfolio weights tensor of shape (batch_size, num_assets).
        :param labels: Label tensor representing asset returns of shape (batch_size, num_assets).
        :return: ``SharpeRatio`` loss tensor, indicating the negative Sharpe Ratio value for the portfolios.
        """
        daily_returns = torch.sum(portfolios * labels, dim=-1)
        # log_returns = torch.log(torch.clamp(daily_returns, min=1e-6))
        excess_returns = daily_returns  # risk_free_rate = 0

        mean_excess_return = torch.mean(excess_returns)
        std_excess_return = torch.std(excess_returns)

        sharpe_ratio = mean_excess_return / (std_excess_return + 1e-6)
        loss = - sharpe_ratio
        return loss

    def compute_volatility_loss(self, portfolios: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute the ``Volatility`` loss, which measures the volatility of the portfolios based on the standard deviation of daily returns.

        This loss function calculates the standard deviation of the daily portfolio returns.

        :param portfolios: Portfolio weights tensor of shape (batch_size, num_assets).
        :param labels: Label tensor representing asset returns of shape (batch_size, num_assets).
        :return: ``Volatility`` loss tensor, representing the volatility of the portfolios based on the standard deviation of daily returns.
        """
        daily_returns = torch.sum(portfolios * labels, dim=-1)
        volatility = torch.std(daily_returns)
        loss = volatility
        return loss

    def compute_custom_criterion_loss(self, portfolios: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute the ``CustomCriterion`` loss,  which is left for the user to define.

        This loss function is a placeholder for the user to implement their own custom loss criterion.

        :param portfolios: Portfolio weights tensor of shape (batch_size, num_assets).
        :param labels: Label tensor representing asset returns of shape (batch_size, num_assets).
        :return: ``CustomCriteria`` loss tensor, representing the user-defined loss criterion.
        """
        # This is a placeholder for the user to implement their own custom loss function.
        # The implementation of the custom loss function is left to the user.
        loss = torch.tensor(0.0, requires_grad=True)
        return loss

    def __call__(self, portfolios: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        criterion_cls = self.criterion_dict.get(self.config["CRITERION_NAME"], None)
        if criterion_cls is None:
            raise ValueError(f"Invalid criterion name: {self.config['CRITERION_NAME']}. Supported criteria are: {self.criterion_dict.keys()}")
        return criterion_cls(portfolios, labels)