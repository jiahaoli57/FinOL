import time
import torch
import numpy as np
import pandas as pd

from finol.utils import load_config, actual_portfolio_selection, add_prefix


class MetricCaculator:
    def __init__(self, load_dataset_output=None, train_model_output=None, mode="normal"):
        self.daily_returns = None
        self.config = load_config()
        self.load_dataset_output = load_dataset_output
        self.train_model_output = train_model_output
        self.mode = mode

        if mode == "normal":
            self.logdir = self.train_model_output["logdir"]
        else:
            self.logdir = None

    def caculate_CW(self):
        # Calculate the Cumulative Wealth (CW)
        self.CW = self.total_returns[-1]

    def caculate_APY(self):
        # Calculate the Annual Percentage Yield (APY)
        num_years = self.num_trading_periods / self.config["NUM_DAYS_PER_YEAR"][self.config["DATASET_NAME"]]  # Number of years
        self.APY = pow(self.total_returns[-1], 1 / num_years) - 1  # APY formula

    def caculate_SR(self):
        # Calculate the Sharpe ratio (SR)
        self.sigma = np.std(self.daily_returns, ddof=1) * pow(self.config["NUM_DAYS_PER_YEAR"][self.config["DATASET_NAME"]], 0.5)  # Standard deviation of daily returns (annualized)
        self.SR = (self.APY - 0.04) / self.sigma  # Sharpe ratio formula, risk-free rate of return, reference CORN in 2011 is 0.04, reference LOAD in 2019 is 0

    def caculate_MDD(self):
        # Calculate the Maximum DrawDown (MDD)
        self.mddvec = []
        self.DMDD = []
        for i in range(self.num_trading_periods):
            j = i + 1
            temp = self.total_returns[:j]
            ddval = (max(temp) - temp[i]) / max(temp)  # Drawdown value for each trading period
            self.mddvec.append(ddval)
            self.DMDD.append(max(self.mddvec))
        self.MDD = max(self.mddvec)  # Maximum drawdown value

    def caculate_ATO(self):
        self.ATO = sum(self.daily_turno_list) / (2 * (self.num_trading_periods))

    def caculate_TCW(self):
        df = pd.DataFrame(columns=["Rate", "TCW"])

        # tc = 0.005
        # interval = 0.001
        for tc_rate in np.arange(0, 0.005 + 0.001, 0.001):
            portfolio_o = np.zeros_like(self.portfolios[0, :])

            self.daily_turno_list = []
            self.daily_return_list = []

            for i in range(self.num_trading_periods):
                label = self.labels[i, :]
                portfolio = self.portfolios[i, :]

                daily_turno = (abs(portfolio_o - portfolio).sum())
                self.daily_turno_list.append(daily_turno)

                daily_return = np.sum(label * portfolio) * (1 - ((tc_rate / 2) * daily_turno))
                portfolio_o = portfolio * label / daily_return

                self.daily_return_list.append(daily_return)

            df.loc[len(df)] = [tc_rate, np.cumprod(self.daily_return_list)[-1]]

        self.TCW = df["TCW"]

    def caculate_DCW(self):
        self.daily_returns = np.sum(self.labels * self.portfolios, axis=1)
        self.total_returns = np.cumprod(self.daily_returns)
        self.DCW = self.total_returns

    def caculate_portfolios(self):
        self.model = torch.load(self.logdir + "/" + add_prefix("best_model.pt")).to(self.config["DEVICE"])
        self.model.eval()
        self.test_loader = self.load_dataset_output["test_loader"]
        self.num_trading_periods = self.load_dataset_output["NUM_TEST_PERIODS"]  # Total number of trading periods
        self.NUM_ASSETS = self.load_dataset_output["NUM_ASSETS"]

        self.portfolios = torch.zeros((self.num_trading_periods, self.NUM_ASSETS))
        self.labels = torch.zeros((self.num_trading_periods, self.NUM_ASSETS))
        start_time = time.time()
        for i, data in enumerate(self.test_loader, 0):
            x_data, label = data
            self.labels[i, :] = label

            final_scores = self.model(x_data.float())
            portfolio = actual_portfolio_selection(final_scores)
            self.portfolios[i, :] = portfolio

        self.RT = time.time() - start_time

        self.portfolios = self.portfolios.cpu().detach().numpy()
        self.labels = self.labels.cpu().detach().numpy()

    def caculate_metric(self, portfolios=None, labels=None, runtime=None):
        if self.mode == "normal":
            self.caculate_portfolios()
        elif self.mode == "ed":
            self.portfolios, self.labels = portfolios.cpu().detach().numpy(), labels.cpu().detach().numpy()
            self.num_trading_periods, self.NUM_ASSETS = portfolios.shape
            self.RT = runtime

        # Profit Metrics
        self.caculate_DCW()
        self.caculate_CW()
        self.caculate_APY()
        self.caculate_SR()
        # Risk Metrics
        self.caculate_MDD()
        # Practical Metrics
        self.caculate_TCW()
        self.caculate_ATO()

        caculate_metric_output = {
            "logdir": self.logdir,
            "CW": self.CW,
            "DCW": self.DCW,
            "APY": self.APY,
            "SR": self.SR,
            "VR": self.sigma,
            "MDD": self.MDD,
            "DDD": self.mddvec,
            "DMDD": self.DMDD,
            "ATO": self.ATO,
            "TCW": self.TCW,
            "RT": self.RT,
        }
        return caculate_metric_output