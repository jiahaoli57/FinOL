import time
import torch
import numpy as np
import pandas as pd

from typing import List, Tuple, Dict, Union, Any
from finol.utils import load_config, actual_portfolio_selection, add_prefix


class MetricCaculator:
    """
    Class to calculate various performance metrics based on the loaded dataset output and trained model output.
    """
    def __init__(self, load_dataset_output=None, train_model_output=None, mode="normal") -> None:
        self.daily_returns = None
        self.config = load_config()
        self.load_dataset_output = load_dataset_output
        self.train_model_output = train_model_output
        self.mode = mode

        if mode == "normal":
            self.logdir = self.train_model_output["logdir"]
        else:
            self.logdir = None

    def caculate_annual_percentage_yield(self, num_trading_periods: float, final_cumulative_wealth: float) -> float:
        """
        *Annualized Percentage Yield* (APY): Annualize the *Final Cumulative Wealth* based on the investment horizon :math:`n` to facilitate comparison across different datasets.

        .. math::

            \mathrm{APY}=\sqrt[y]{\mathrm{FCW}}-1,

        where :math:`y` is the number of years in the investment period.

        :param num_trading_periods: Number of trading periods :math:`n`.
        :param final_cumulative_wealth: *Final Cumulative Wealth*.
        :return: *Annual Percentage Yield*.
        """
        num_years = num_trading_periods / self.config["NUM_DAYS_PER_YEAR"][self.config["DATASET_NAME"]]  # Number of years
        annual_percentage_yield = pow(final_cumulative_wealth, 1 / num_years) - 1  # APY formula
        return annual_percentage_yield

    def caculate_sharpe_ratio(self, daily_returns: np.ndarray, annual_percentage_yield: float) -> float:
        """
        *Sharpe Ratio* (SR): Measure risk-adjusted return using the portfolio's excess return over risk-free rate per unit of volatility risk.

        .. math::

            \mathrm{SR}=\\frac{\mathrm{APY}-R^f}{\sigma_n},

        where :math:`R^f = 0.04` is the annualized risk-free rate and :math:`\sigma_n` denotes the annualized standard deviation of portfolio daily returns.

        :param daily_returns: Daily return sequence.
        :param annual_percentage_yield: *Annual Percentage Yield*.
        :return: *Sharpe Ratio*.
        """
        sigma = np.std(daily_returns, ddof=1) * pow(self.config["NUM_DAYS_PER_YEAR"][self.config["DATASET_NAME"]], 0.5)  # Standard deviation of daily returns (annualized)
        sharpe_ratio = (annual_percentage_yield - 0.04) / sigma  # Sharpe ratio formula, risk-free rate of return, reference CORN in 2011 is 0.04, reference LOAD in 2019 is 0
        return sharpe_ratio

    def caculate_volatility_risk(self, daily_returns) -> float:
        """
        *Volatility Risk* (VR): Calculate the annualized standard deviation :math:`\sigma_n` of daily portfolio returns, indicating return fluctuation risk.

        :param daily_returns: Daily return sequence.
        :return: *Volatility Risk*.
        """
        sigma = np.std(daily_returns, ddof=1) * pow(self.config["NUM_DAYS_PER_YEAR"][self.config["DATASET_NAME"]], 0.5)  # Standard deviation of daily returns (annualized)
        return sigma

    def caculate_maximum_drawdown(self, daily_cumulative_wealth) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        *Maximum DrawDown* (MDD): Measure the maximum peak-to-trough decline of a portfolio over the investment horizon :math:`n`.

        .. math::

            \mathrm{MDD}=\\max_{0\leq t\leq n}\\left(\\frac{\\max_{0\leq k\leq t} S_k - S_t}{\\max_{0\leq k\leq t} S_k}\\right),

        where :math:`S_t` is the cumulative wealth at the end of the :math:`t`-th trading period.

        :param daily_cumulative_wealth: *Daily Cumulative Wealth*.
        :return: Tuple containing the daily drawdown, daily maximum drawdown, and maximum drawdown.
        """
        daily_drawdown = []
        daily_maximum_drawdown = []
        num_trading_periods = len(daily_cumulative_wealth)
        for i in range(num_trading_periods):
            j = i + 1
            temp = daily_cumulative_wealth[:j]
            ddval = (max(temp) - temp[i]) / max(temp)  # Drawdown value for each trading period
            daily_drawdown.append(ddval)
            daily_maximum_drawdown.append(max(daily_drawdown))
        maximum_drawdown = max(daily_drawdown)  # Maximum drawdown value

        # Convert lists to NumPy arrays
        daily_drawdown = np.array(daily_drawdown)
        daily_maximum_drawdown = np.array(daily_maximum_drawdown)

        return daily_drawdown, daily_maximum_drawdown, maximum_drawdown

    def caculate_average_turnover(self, daily_turnovers) -> None:
        """
        *Average Turnover* (ATO): Calculate the average proportion of portfolio vectors changed between periods, indicating rebalancing frequency.

        .. math::

            \mathrm{ATO}=\\frac{1}{2(n)} \\sum_{t=1}^n\\left\|\mathbf{b}_t-\\hat{\mathbf{b}}_{t-1}\\right\|_1,

        where portfolio :math:`\\hat{\mathbf{b}}_{t-1} = (\\frac{b_{t-1, 1}  x_{t-1, 1}}{\mathbf{b}_{t-1}^{\\top} \mathbf{x}_{t-1}}, \\ldots, \\frac{b_{t-1, m}  x_{t-1, m}}{\mathbf{b}_{t-1}^{\\top} \mathbf{x}_{t-1}})` is the portfolio vector at the end of :math:`(t-1)`-th period adjusted for asset returns. For simplicity, we set all elements of :math:`\hat{\mathbf{b}}_{0}` to 0.

        :param daily_turnovers: Daily turnover sequence.
        :return: *Average Turnover*
        """
        num_trading_periods = len(daily_turnovers)
        average_turnover = sum(daily_turnovers) / (2 * num_trading_periods)
        return average_turnover

    def caculate_transaction_costs_wealth(self, portfolios, labels) -> Tuple[np.ndarray, pd.Series]:
        """
        *Transaction Costs-Adjusted Cumulative Wealth* (TCW):  Apply proportional transaction costs to the raw *Cumulative Wealth* equation based on rebalancing.

        .. math::

            \mathrm{TCW} = S_0 \\prod_{t=1}^n\\left[\\left(\mathbf{b}_t^{\\top}  \mathbf{x}_t\\right) \\times\\left(1-\\frac{c}{2} \\times \\left\|\mathbf{b}_t-\\hat{\mathbf{b}}_{t-1}\\right\|_1\\right)\\right],

        where :math:`S_0` is the initial cumulative wealth, :math:`\mathbf{b}_t` is the portfolio vector and :math:`\mathbf{x}_t` is the price relative vector in period :math:`t`. :math:`c \in (0,1)` is the transaction costs rate.

        :param portfolios: Portfolio sequence.
        :param labels: Price relative sequence.
        :return: Tuple containing daily turnover and *transaction costs-adjusted cumulative wealth*.
        """
        num_trading_periods, num_assets = portfolios.shape

        df = pd.DataFrame(columns=["Rate", "TCW"])

        # tc = 0.005
        # interval = 0.001
        for tc_rate in np.arange(0, 0.005 + 0.001, 0.001):
            portfolio_o = np.zeros_like(portfolios[0, :])

            daily_turnover = []
            daily_returns = []

            for i in range(num_trading_periods):
                label = labels[i, :]
                portfolio = portfolios[i, :]

                turnover = (abs(portfolio_o - portfolio).sum())
                daily_return = np.sum(label * portfolio) * (1 - ((tc_rate / 2) * turnover))
                portfolio_o = portfolio * label / daily_return

                daily_turnover.append(turnover)
                daily_returns.append(daily_return)

            df.loc[len(df)] = [tc_rate, np.cumprod(daily_returns)[-1]]

        daily_turnovers = np.array(daily_turnover)
        transaction_costs_wealth = df["TCW"]
        return daily_turnovers, transaction_costs_wealth


    def caculate_final_cumulative_wealth(self, portfolios: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        *Final Cumulative Wealth* (FCW): Calculate the final cumulative wealth generated by the portfolio sequence :math:`\mathbf{b}_{1:n}` at the end of the :math:`n`-th trading periods.

        .. math::

            \mathrm{FCW}=S_0 \prod_{t=1}^n \mathbf{b}_t^{\\top} \mathbf{x}_t=S_0 \prod_{t=1}^n \sum_{i=1}^m b_{t, i} x_{t, i},

        where :math:`S_0` is the initial cumulative wealth, :math:`\mathbf{b}_t` is the portfolio vector and :math:`\mathbf{x}_t` is the price relative vector in period :math:`t`.

        :param portfolios: Portfolio sequence :math:`\mathbf{b}_{1:n} \in \mathbb{R}^{n {\\times} m}`.
        :param labels: Price relative sequence :math:`\mathbf{x}_{1:n} \in \mathbb{R}_{+}^{n {\\times} m}`.
        :return: Tuple containing the daily return, daily cumulative wealth, and final cumulative wealth.
        """
        # Calculate daily returns
        daily_returns = np.sum(portfolios * labels, axis=1)
        # Calculate daily cumulative wealth
        daily_cumulative_wealth = np.cumprod(daily_returns)
        # Calculate final cumulative wealth
        final_cumulative_wealth = daily_cumulative_wealth[-1]

        return daily_returns, daily_cumulative_wealth, final_cumulative_wealth

    def caculate_portfolios(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Calculate the portfolio sequence :math:`\mathbf{b}_{1:n}`.

        :return: Tuple containing the portfolio sequence :math:`\mathbf{b}_{1:n}`, price relative sequence :math:`\mathbf{x}_{1:n}`, and other information.
        """
        model = torch.load(self.logdir + "/" + add_prefix("best_model.pt")).to(self.config["DEVICE"])
        model.eval()
        test_loader = self.load_dataset_output["test_loader"]
        num_trading_periods = self.load_dataset_output["num_test_periods"]  # Total number of trading periods
        num_assets = self.load_dataset_output["num_assets"]

        portfolios = torch.zeros((num_trading_periods, num_assets))
        labels = torch.zeros((num_trading_periods, num_assets))
        start_time = time.time()
        for i, data in enumerate(test_loader, 0):
            x_data, label = data
            labels[i, :] = label

            final_scores = model(x_data.float())
            portfolio = actual_portfolio_selection(final_scores)
            portfolios[i, :] = portfolio

        runtime = time.time() - start_time
        portfolios = portfolios.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()

        return portfolios, labels, runtime, num_trading_periods

    def caculate_runtime(self, runtime: float) -> float:
        """
        *Running Time* (RT): Measure the amount of time required to execute trades for an OLPS method. In live markets, any delays in processing or slow computation times can result in missed opportunities or executions at less favorable prices.

        .. note::

            While it is typically assumed in OLPS problems that execution at the desired price is always possible.

        :param runtime: *Running Time*.
        :return: *Running Time*.
        """
        return runtime

    def caculate_metric(self, portfolios=None, labels=None, runtime=None) -> Dict:
        """
        Calculate various performance metrics based on the provided portfolios, labels.

        :param portfolios: Portfolio sequence :math:`\mathbf{b}_{1:n}`.
        :param labels: Price relative sequence :math:`\mathbf{x}_{1:n}`.
        :param runtime: The runtime information. Required only when ``self.mode`` is ``ed`` (economic distillation).
        :return: Dictionary containing calculated performance metrics.
        """
        if self.mode == "normal":
            portfolios, labels, runtime, num_trading_periods = self.caculate_portfolios()
        elif self.mode == "ed":
            portfolios, labels = portfolios.cpu().detach().numpy(), labels.cpu().detach().numpy()
            num_trading_periods, num_assets = portfolios.shape

        # Profit Metrics
        daily_returns, daily_cumulative_wealth, final_cumulative_wealth = self.caculate_final_cumulative_wealth(portfolios, labels)
        annual_percentage_yield = self.caculate_annual_percentage_yield(num_trading_periods, final_cumulative_wealth)
        sharpe_ratio = self.caculate_sharpe_ratio(daily_returns, annual_percentage_yield)

        # Risk Metrics
        volatility_risk = self.caculate_volatility_risk(daily_returns)
        daily_drawdown, daily_maximum_drawdown, maximum_drawdown = self.caculate_maximum_drawdown(daily_cumulative_wealth)

        # Practical Metrics
        daily_turnover, transaction_costs_wealth = self.caculate_transaction_costs_wealth(portfolios, labels)
        average_turnover = self.caculate_average_turnover(daily_turnover)
        runtime = self.caculate_runtime(runtime)

        caculate_metric_output = {
            "logdir": self.logdir,
            "DCW": daily_cumulative_wealth,
            "CW": final_cumulative_wealth,
            "APY": annual_percentage_yield,
            "SR": sharpe_ratio,
            "VR": volatility_risk,
            "DDD": daily_drawdown,
            "DMDD": daily_maximum_drawdown,
            "MDD": maximum_drawdown,
            "ATO": average_turnover,
            "TCW": transaction_costs_wealth,
            "RT": runtime,
        }
        return caculate_metric_output