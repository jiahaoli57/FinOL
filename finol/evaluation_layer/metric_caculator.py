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

    for tc_rate in np.arange(0, TRANSACTIOS_COSTS_RATE + TRANSACTIOS_COSTS_RATE_INTERVAL, TRANSACTIOS_COSTS_RATE_INTERVAL):
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
    return df["TCW"]


def caculate_metric(train_model_output, load_dataset_output):
    model = train_model_output["best_model"]
    logdir = train_model_output["logdir"]
    if model == None:
        model = torch.load(logdir + '/best_model_'+DATASET_NAME+'.pt')

    model.eval()
    test_loader = load_dataset_output["test_loader"]
    NUM_TEST_PERIODS = load_dataset_output["NUM_TEST_PERIODS"]
    NUM_ASSETS = load_dataset_output["NUM_ASSETS"]

    portfolios = torch.zeros((NUM_TEST_PERIODS, NUM_ASSETS))
    labels = torch.zeros((NUM_TEST_PERIODS, NUM_ASSETS))

    start_time = time.time()
    for i, data in enumerate(test_loader, 0):
        x_data, label = data
        labels[i, :] = label

        portfolio = model(x_data.float())
        portfolios[i, :] = portfolio

    runtime = time.time() - start_time

    labels = labels.cpu().detach().numpy()
    portfolios = portfolios.cpu().detach().numpy()

    daily_return = np.sum(labels * portfolios, axis=1)
    total_return = np.cumprod(daily_return)

    caculate_metric_output = {}

    if METRIC_CONFIG.get('INCLUDE_PROFIT_METRICS'):
        caculate_metric_output["CW"] = caculate_CW(total_return)
        caculate_metric_output["DCW"] = total_return
        caculate_metric_output["APY"] = caculate_APY(total_return)
        caculate_metric_output["SR"], sigma = caculate_SR(daily_return, caculate_metric_output["APY"])

    if METRIC_CONFIG.get('INCLUDE_RISK_METRICS'):
        caculate_metric_output["VR"] = sigma
        caculate_metric_output["MDD"], caculate_metric_output["DDD"] = caculate_MDD(total_return)

    if METRIC_CONFIG.get("PRACTICAL_METRICS")["INCLUDE_PRACTICAL_METRICS"]:
        caculate_metric_output["ATO"] = caculate_ATO(NUM_TEST_PERIODS, NUM_ASSETS, model, test_loader)
        caculate_metric_output["TCW"] = caculate_TCW(NUM_TEST_PERIODS, NUM_ASSETS, model, test_loader)
        caculate_metric_output["RT"] = runtime

    caculate_metric_output["logdir"] = logdir
    return caculate_metric_output