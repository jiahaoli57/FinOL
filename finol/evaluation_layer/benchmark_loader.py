import argparse
import json5
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from rich import print
from finol import utils
from finol.config import *

def get_variable_name(var):
    return next(key for key, value in globals().items() if value is var)


def plot_dataframe(df, column_names, plot_type):
    plt.style.use('seaborn-talk')  # fivethirtyeight bmh seaborn-talk seaborn-poster seaborn-white
    plt.figure(figsize=(9, 4))

    markevery = {
        'NYSE(O)': 50,
        'NYSE(N)': 55,
        'DJIA': 3,
        'SP500': 7,
        'TSE': 6,
        'SSE': 4,
        'HSI': 4,
        'CMEG': 4,
        'CRYPTO': 13
    }.get(DATASET_NAME, 10)

    if plot_type == 'DCW':
        xlabel = 'Trading Periods'
        ylabel = 'Daily Cumulative Wealth'
    elif plot_type == 'DDD':
        xlabel = 'Trading Periods'
        ylabel = 'Daily DrawDown'
    elif plot_type == 'TCW':
        markevery = 1
        xlabel = 'Transaction Costs Rates (%)'
        ylabel = 'Transaction Costs-Adjusted Cumulative Wealth'
        plt.xticks(ticks=[0, 2, 4, 6, 8, 10], labels=[0, 0.2, 0.4, 0.6, 0.8, 1])

    df = df[column_names]
    num_columns = len(df.columns)

    # colors = ['black'] * (num_columns - 1) + ['red']
    colors = ['black'] * (num_columns)
    lines = ['-'] * (num_columns - 1) + [':']
    markers = ['o', '^', '<', '>', 's', 'p', 'h', '+', 'x', '|', '_']

    for i, column in enumerate(df.columns):
        # plt.plot(df[column], color=colors[i], marker=markers[i], markevery=markevery, markeredgecolor=colors[i],
        #          markerfacecolor='none', alpha=0.6, label=column, linewidth=2)
        plt.plot(df[column], linestyle=lines[i], color=colors[i], marker=markers[i], markevery=markevery,  alpha=0.5, label=column)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(DATASET_NAME)
    plt.legend(loc='best')  # lower left
    plt.grid(True)
    # plt.yscale('log')
    plt.tight_layout()
    plt.savefig(DATASET_NAME + '_' + get_variable_name(column_names) + '_' + plot_type + '.pdf',
                format='pdf',
                dpi=300,
                bbox_inches='tight')
    plt.show()


def load_benchmark(caculate_metric_output=None):
    """
    Load baseline results from a CSV file for specified target classes.

    Returns:
    - None
    """
    if DATASET_NAME == 'NYSE(O)':
        _ = "1"
    elif DATASET_NAME == 'NYSE(N)':
        _ = "2"
    elif DATASET_NAME == 'DJIA':
        _ = "3"
    elif DATASET_NAME == 'SP500':
        _ = "4"
    elif DATASET_NAME == 'TSE':
        _ = '5'
    elif DATASET_NAME in ['SSE']:
        _ = '6'
    elif DATASET_NAME in ['HSI']:
        _ = '7'
    elif DATASET_NAME in ['CMEG']:
        _ = '8'
    elif DATASET_NAME in ['CRYPTO']:
        _ = '9'

    print(caculate_metric_output)
    if METRIC_CONFIG.get('INCLUDE_PROFIT_METRICS'):
        daily_return = pd.read_excel(ROOT_PATH + r'/benchmark_results/profit_metrics/'+_+'/daily_return.xlsx')
        daily_cumulative_wealth = pd.read_excel(ROOT_PATH + r'/benchmark_results/profit_metrics/'+_+'/daily_cumulative_wealth.xlsx')
        final_profit_result = pd.read_excel(ROOT_PATH + r'/benchmark_results/profit_metrics/'+_+'/final_profit_result.xlsx')

        daily_return = daily_return.dropna(axis=1, how="any")
        daily_cumulative_wealth = daily_cumulative_wealth.dropna(axis=1, how="any")
        final_profit_result = final_profit_result.dropna(axis=1, how="any")
        if caculate_metric_output != None:
            daily_cumulative_wealth[MODEL_NAME] = caculate_metric_output["DCW"]
            final_profit_result = pd.concat([final_profit_result, [caculate_metric_output["CW"], caculate_metric_output["APY"], caculate_metric_output["SR"]]], axis=1)
        print(
            final_profit_result
        )
        plot_dataframe(daily_cumulative_wealth, PLOT_ALL_1, 'DCW')
        plot_dataframe(daily_cumulative_wealth, PLOT_ALL_2, 'DCW')

    if METRIC_CONFIG.get('INCLUDE_RISK_METRICS'):
        daily_drawdown = pd.read_excel(ROOT_PATH + r'/benchmark_results/risk_metrics/'+_+'/daily_drawdown.xlsx')
        final_risk_result = pd.read_excel(ROOT_PATH + r'/benchmark_results/risk_metrics/'+_+'/final_risk_result.xlsx')

        daily_drawdown = daily_drawdown.dropna(axis=1, how="any")
        final_risk_result = final_risk_result.dropna(axis=1, how="any")
        if caculate_metric_output != None:
            daily_drawdown[MODEL_NAME] = caculate_metric_output["DDD"]
            final_risk_result = pd.concat([final_risk_result, [caculate_metric_output["VR"], caculate_metric_output["MDD"]]], axis=1)
        print(
            final_risk_result
        )
        plot_dataframe(daily_drawdown, PLOT_ALL_1, 'DDD')
        plot_dataframe(daily_drawdown, PLOT_ALL_2, 'DDD')

    if METRIC_CONFIG.get('PRACTICAL_METRICS')['INCLUDE_PRACTICAL_METRICS']:
        transaction_costs_adjusted_cumulative_wealth = pd.read_excel(ROOT_PATH + r'/benchmark_results/practical_metrics/'+_+'/transaction_costs_adjusted_cumulative_wealth.xlsx')
        final_practical_result = pd.read_excel(ROOT_PATH + r'/benchmark_results/practical_metrics/'+_+'/final_practical_result.xlsx')

        transaction_costs_adjusted_cumulative_wealth = transaction_costs_adjusted_cumulative_wealth.dropna(axis=1, how="any")
        final_practical_result = final_practical_result.dropna(axis=1, how="any")
        if caculate_metric_output != None:
            transaction_costs_adjusted_cumulative_wealth[MODEL_NAME] = caculate_metric_output["TCW"]
            final_practical_result = pd.concat([final_practical_result, [caculate_metric_output["ATO"], caculate_metric_output["RT"]]], axis=1)
        print(
            final_practical_result
        )
        plot_dataframe(transaction_costs_adjusted_cumulative_wealth, PLOT_ALL_1, 'TCW')
        plot_dataframe(transaction_costs_adjusted_cumulative_wealth, PLOT_ALL_2, 'TCW')



if __name__ == '__main__':
    # utils.check_update()
    load_benchmark()