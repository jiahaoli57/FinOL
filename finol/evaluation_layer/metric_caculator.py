import time

import torch
import numpy as np
import pandas as pd
from finol.config import *

if DATASET_NAME in ["SSE", "HSI", "CMEG"]:
    day = 52
else:
    day = 252


def caculate_CW(total_return):
    # Calculate the Cumulative Wealth (CW)
    return total_return[-1]


def caculate_APY(total_return):
    # Calculate the Annual Percentage Yield (APY)
    T = len(total_return)  # Total number of trading periods
    y = T / day  # Number of years
    APY = pow(total_return[-1], 1 / y) - 1  # APY formula
    return APY


def caculate_SR(daily_return, APY):
    # Calculate the Sharpe ratio (SR)
    sigma = np.std(daily_return, ddof=1) * pow(day, 0.5)  # Standard deviation of daily returns (annualized)
    r = 0.04  # Risk-free rate of return, reference CORN in 2011 is 0.04, reference LOAD in 2019 is 0
    SR = (APY - r) / sigma  # Sharpe ratio formula
    return SR, sigma


def caculate_MDD(total_return):
    # Calculate the Maximum DrawDown (MDD)
    mddvec = []
    T = len(total_return)
    for i in range(T):
        j = i + 1
        temp = total_return[:j]
        ddval = (max(temp) - temp[i]) / max(temp)  # Drawdown value for each trading period
        mddvec.append(ddval)
    MDD = max(mddvec)  # Maximum drawdown value
    return MDD, mddvec

def caculate_ATO(NUM_PERIODS, NUM_ASSETS, model, test_loader):
    portfolio_o = np.zeros(NUM_ASSETS)
    label_list = []
    portfolio_list = []
    daily_turno_list = []
    # start_time = time.time()
    for i, data in enumerate(test_loader, 1):
        x_data, label = data
        label = label.cpu().detach().numpy()
        label_list.append(label)

        portfolio = model(x_data.float())
        portfolio = portfolio.cpu().detach().numpy()
        portfolio_list.append(portfolio)

        daily_turno = (abs(portfolio_o - portfolio).sum())
        daily_turno_list.append(daily_turno)

        r_0 = np.sum(label * portfolio) * (1 - ((0 / 2) * daily_turno))
        portfolio_o = portfolio * label / r_0

    ATO = sum(daily_turno_list) / (2 * (NUM_PERIODS))
    return ATO

def caculate_TCW(NUM_PERIODS, NUM_ASSETS, model, test_loader):
    df = pd.DataFrame(columns=['Rate', 'TCW'])

    TRANSACTIOS_COSTS_RATE = METRIC_CONFIG.get("PRACTICAL_METRICS")["TRANSACTIOS_COSTS_RATE"]
    TRANSACTIOS_COSTS_RATE_INTERVAL = METRIC_CONFIG.get("PRACTICAL_METRICS")["TRANSACTIOS_COSTS_RATE_INTERVAL"]

    for tc_rate in np.arange(0, TRANSACTIOS_COSTS_RATE, TRANSACTIOS_COSTS_RATE_INTERVAL):
        portfolio_o = np.zeros(NUM_ASSETS)
        label_list = []
        portfolio_list = []
        daily_turno_list = []
        daily_return_list = []
        # start_time = time.time()
        for i, data in enumerate(test_loader, 1):
            x_data, label = data
            label = label.cpu().detach().numpy()
            label_list.append(label)

            portfolio = model(x_data.float())
            portfolio = portfolio.cpu().detach().numpy()
            portfolio_list.append(portfolio)

            daily_turno = (abs(portfolio_o - portfolio).sum())
            daily_turno_list.append(daily_turno)

            r_0 = np.sum(label * portfolio) * (1 - ((tc_rate / 2) * daily_turno))
            portfolio_o = portfolio * label / r_0

            daily_return_list.append(r_0)

        result = np.cumprod(daily_return_list)
        df.loc[len(df)] = [tc_rate, result[-1]]
    return df


def caculate_metric(model, test_loader):
    label_list = []
    portfolio_list = []

    start_time = time.time()
    for i, data in enumerate(test_loader, 1):
        x_data, label = data
        label_list.append(label)

        portfolio = model(x_data.float())
        portfolio_list.append(portfolio)

    runtime = time.time() - start_time
    labels = torch.cat(label_list, dim=0)
    portfolios = torch.cat(portfolio_list, dim=0)

    CW = None
    DCW = None
    APY = None
    SR = None
    VR = None
    ATO = None
    TCW = None
    RT = None
    labels = labels.cpu().detach().numpy()
    portfolios = portfolios.cpu().detach().numpy()
    NUM_PERIODS = portfolios.shape[0]
    NUM_ASSETS = portfolios.shape[1]

    daily_return = np.sum(labels * portfolios, axis=1)
    total_return = np.cumprod(daily_return)

    if METRIC_CONFIG.get('INCLUDE_PROFIT_METRICS'):
        CW = caculate_CW(total_return)
        DCW = total_return
        APY = caculate_APY(total_return)
        SR, sigma = caculate_SR(daily_return, APY)

    if METRIC_CONFIG.get('INCLUDE_RISK_METRICS'):
        VR = sigma
        MDD, DDD = caculate_MDD(total_return)

    if METRIC_CONFIG.get("PRACTICAL_METRICS")["INCLUDE_PRACTICAL_METRICS"]:
        ATO = caculate_ATO(NUM_PERIODS, NUM_ASSETS, model, test_loader)
        TCW = caculate_TCW(NUM_PERIODS, NUM_ASSETS, model, test_loader)
        RT = runtime


    caculate_metric_output = {
        "CW": CW,
        "DCW": DCW,
        "APY": APY,
        "SR": SR,
        "VR": VR,
        "MDD": MDD,
        "DDD": DDD,
        "ATO": ATO,
        "TCW": TCW,
        "RT": RT,
    }
    return caculate_metric_output
# # Calculate some numerical results based on the rewards and other metrics
#
# daily_return = all_reward_list
# total_return = test_wealth_list
#
# # Calculate and print the average turnover (AT) rate
# print(f'Final AT rate: {sum(dai_TO) / (2 * (len(dai_TO) - 1))}')
#
#
# day = 252  # Number of trading days in a year
# T = len(dai_TO)  # Total number of trading periods
# y = T / day  # Number of years
#
# APY = pow(total_return[-1], 1 / y) - 1  # APY formula
# print(f'y: {y}')
# print(f'APY: {APY}')
#

#

#
# # Calculate and print the volatility ratio (VR)
# VR = sigma  # Volatility ratio is the same as standard deviation of daily returns (annualized)
# print(f'VR: {VR}')
#
# # Create a table to display the numerical results using rich.table module
# table = Table(title="Numerical results")
#
# table.add_column("Metrics", style="cyan")  # no_wrap=True
# table.add_column("Results", style="magenta")
#
# table.add_row("Return", f'&{round(test_wealth, 2)}&{round(APY, 2)}&{round(SR, 2)}')
# table.add_row("Risk", f'&{round(sigma, 2)}&{round(MDD, 2)}')
# table.add_row("TO", f'&{round((sum(dai_TO) / (2 * (len(dai_TO) - 1))) * 100, 2)}\%')
# table.add_row("Ablation Study", f'&{round(test_wealth, 2)}&{round(SR, 2)}&{round(MDD, 2)}')
#
# console = Console()
# console.print(table)
