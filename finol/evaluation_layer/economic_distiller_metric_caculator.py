import time
import torch
import numpy as np
import pandas as pd

from finol.config import *


if DATASET_NAME in ["SSE", "HSI", "CMEG", "Nasdaq-100"]:
    day = 52
else:
    day = 252


def ed_caculate_CW(total_return):
    # Calculate the Cumulative Wealth (CW)
    return total_return[-1]


def ed_caculate_APY(total_return):
    # Calculate the Annual Percentage Yield (APY)
    T = len(total_return)  # Total number of trading periods
    y = T / day  # Number of years
    APY = pow(total_return[-1], 1 / y) - 1  # APY formula
    return APY


def ed_caculate_SR(daily_return, APY):
    # Calculate the Sharpe ratio (SR)
    sigma = np.std(daily_return, ddof=1) * pow(day, 0.5)  # Standard deviation of daily returns (annualized)
    r = 0.04  # Risk-free rate of return, reference CORN in 2011 is 0.04, reference LOAD in 2019 is 0
    SR = (APY - r) / sigma  # Sharpe ratio formula
    return SR, sigma


def ed_caculate_MDD(total_return):
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


def ed_caculate_ATO(portfolios, labels):
    NUM_TEST_PERIODS, NUM_ASSETS = portfolios.shape
    portfolio_o = np.zeros(NUM_ASSETS)
    daily_turno_list = []
    for day in range(NUM_TEST_PERIODS):
        label = labels[day, :]
        portfolio = portfolios[day, :]

        daily_turno = (abs(portfolio_o - portfolio).sum())
        daily_turno_list.append(daily_turno)

        daily_return = np.sum(label * portfolio) * (1 - ((0 / 2) * daily_turno))
        portfolio_o = portfolio * label / daily_return

    ATO = sum(daily_turno_list) / (2 * (NUM_TEST_PERIODS))
    return ATO


def ed_caculate_TCW(portfolios, labels):
    NUM_TEST_PERIODS, NUM_ASSETS = portfolios.shape
    df = pd.DataFrame(columns=['Rate', 'TCW'])

    TRANSACTIOS_COSTS_RATE = METRIC_CONFIG.get("PRACTICAL_METRICS")["TRANSACTIOS_COSTS_RATE"]
    TRANSACTIOS_COSTS_RATE_INTERVAL = METRIC_CONFIG.get("PRACTICAL_METRICS")["TRANSACTIOS_COSTS_RATE_INTERVAL"]

    for tc_rate in np.arange(0, TRANSACTIOS_COSTS_RATE + TRANSACTIOS_COSTS_RATE_INTERVAL, TRANSACTIOS_COSTS_RATE_INTERVAL):
        portfolio_o = np.zeros(NUM_ASSETS)
        daily_turno_list = []
        daily_return_list = []

        for day in range(NUM_TEST_PERIODS):
            label = labels[day, :]
            portfolio = portfolios[day, :]

            daily_turno = (abs(portfolio_o - portfolio).sum())
            daily_turno_list.append(daily_turno)

            daily_return = np.sum(label * portfolio) * (1 - ((tc_rate / 2) * daily_turno))
            portfolio_o = portfolio * label / daily_return

            daily_return_list.append(daily_return)

        result = np.cumprod(daily_return_list)
        df.loc[len(df)] = [tc_rate, result[-1]]
    return df["TCW"]


def ed_caculate_metric(portfolios, labels, runtime):
    portfolios = portfolios.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()

    daily_return = np.sum(labels * portfolios, axis=1)
    total_return = np.cumprod(daily_return)

    economic_distiller_caculate_metric_output = {}

    if METRIC_CONFIG.get('INCLUDE_PROFIT_METRICS'):
        economic_distiller_caculate_metric_output["CW"] = ed_caculate_CW(total_return)
        economic_distiller_caculate_metric_output["DCW"] = total_return
        economic_distiller_caculate_metric_output["APY"] = ed_caculate_APY(total_return)
        economic_distiller_caculate_metric_output["SR"], sigma = ed_caculate_SR(daily_return, economic_distiller_caculate_metric_output["APY"])

    if METRIC_CONFIG.get('INCLUDE_RISK_METRICS'):
        economic_distiller_caculate_metric_output["VR"] = sigma
        economic_distiller_caculate_metric_output["MDD"], economic_distiller_caculate_metric_output["DDD"] = ed_caculate_MDD(total_return)

    if METRIC_CONFIG.get("PRACTICAL_METRICS")["INCLUDE_PRACTICAL_METRICS"]:
        economic_distiller_caculate_metric_output["ATO"] = ed_caculate_ATO(portfolios, labels)
        economic_distiller_caculate_metric_output["TCW"] = ed_caculate_TCW(portfolios, labels)
        economic_distiller_caculate_metric_output["RT"] = runtime

    # caculate_metric_output["logdir"] = logdir
    return economic_distiller_caculate_metric_output