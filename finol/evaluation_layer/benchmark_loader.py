import pandas as pd

from finol.config import *


def get_variable_name(var):
    return next(key for key, value in globals().items() if value is var)


def plot_dataframe(df, column_names, plot_type, logdir):
    fig = plt.figure(figsize=(9, 4))

    if plot_type == 'DCW':
        markevery = MARKEVERY
        if CHINESE_PLOT:
            xlabel = '交易期'
            ylabel = '逐期累积财富'
        else:
            xlabel = 'Trading Periods'
            ylabel = 'Daily Cumulative Wealth'

    elif plot_type == 'DDD':
        markevery = MARKEVERY
        if CHINESE_PLOT:
            xlabel = '交易期'
            ylabel = '逐期下行风险'
        else:
            xlabel = 'Trading Periods'
            ylabel = 'Daily DrawDown'

    elif plot_type == 'TCW':
        plt.figure(figsize=(6, 5))  # Normal size
        TRANSACTIOS_COSTS_RATE = METRIC_CONFIG.get("PRACTICAL_METRICS")["TRANSACTIOS_COSTS_RATE"]
        TRANSACTIOS_COSTS_RATE_INTERVAL = METRIC_CONFIG.get("PRACTICAL_METRICS")["TRANSACTIOS_COSTS_RATE_INTERVAL"]
        markevery = 1
        if CHINESE_PLOT:
            xlabel = '交易费用率 (%)'
            ylabel = '考虑交易费用的累积财富'
        else:
            xlabel = 'Transaction Costs Rates (%)'
            ylabel = 'Costs-Adjusted Cumulative Wealth'

        plt.xticks(ticks=[0, 2, 4, 6, 8, 10], labels=[0, 0.2, 0.4, 0.6, 0.8, 1])
        plt.xticks(ticks=[0, 2.5, 5, 7.5, 10], labels=[0, 0.25, 0.5, 0.75, 1])
        plt.xticks(ticks=[0, 1.25, 2.5, 3.75, 5, 7.5, 10], labels=[0, 0.125, 0.25, 0.375, 0.5, 0.75, 1])

    df = df[column_names]
    # df = df.rename(columns={df.columns[-2]: 'Teacher-Model', df.columns[-1]: 'Student-Model'})
    num_columns = len(df.columns)

    if INTERPRETABLE_ANALYSIS_CONFIG['INCLUDE_ECONOMIC_DISTILLATION']:
        colors = ['black'] * (num_columns - 2) + ['red'] * 2
    else:
        colors = ['black'] * (num_columns - 1) + ['red'] * 1

    lines = ['-'] * (num_columns - 1) + [':']

    if plot_type == 'TCW':
        for i, column in enumerate(df.columns):
            plt.plot(df[column].head(int(TRANSACTIOS_COSTS_RATE/TRANSACTIOS_COSTS_RATE_INTERVAL)+1), linestyle=lines[i], color=colors[i], marker=MARKERS[i], markevery=markevery, alpha=0.5,
                     label=column)
    else:
        for i, column in enumerate(df.columns):
            plt.plot(df[column], linestyle=lines[i], color=colors[i], marker=MARKERS[i], markevery=markevery, alpha=0.5,
                     label=column)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(DATASET_NAME)
    plt.legend(loc='best')  # lower left  upper left
    plt.grid(True)
    # plt.yscale('log')
    plt.tight_layout()
    plt.savefig(logdir + '/' + MODEL_NAME + '_' + DATASET_NAME + '_' + get_variable_name(column_names) + '_' + plot_type + '.pdf',
                format='pdf',
                dpi=300,
                bbox_inches='tight')
    plt.show()


def load_benchmark(caculate_metric_output, economic_distiller_caculate_metric_output):
    """
    Load baseline results from a CSV file for specified target classes.

    Returns:
    - None
    """
    if caculate_metric_output != None:
        logdir = caculate_metric_output['logdir']
    else:
        logdir = ROOT_PATH

    if METRIC_CONFIG.get('INCLUDE_PROFIT_METRICS'):
        daily_return = pd.read_excel(ROOT_PATH + '/data/benchmark_results/profit_metrics/'+DATASET_NAME+'/daily_return.xlsx')
        daily_cumulative_wealth = pd.read_excel(ROOT_PATH + '/data/benchmark_results/profit_metrics/'+DATASET_NAME+'/daily_cumulative_wealth.xlsx')
        final_profit_result = pd.read_excel(ROOT_PATH + '/data/benchmark_results/profit_metrics/'+DATASET_NAME+'/final_profit_result.xlsx')

        daily_return = daily_return.dropna(axis=1, how="any")
        daily_cumulative_wealth = daily_cumulative_wealth.dropna(axis=1, how="any")
        final_profit_result = final_profit_result.dropna(axis=1, how="any")
        if caculate_metric_output != None:
            daily_cumulative_wealth[MODEL_NAME] = caculate_metric_output["DCW"]
            final_profit_result = final_profit_result.assign(**{MODEL_NAME: 0})
            final_profit_result.loc[0, MODEL_NAME] = caculate_metric_output["CW"]
            final_profit_result.loc[1, MODEL_NAME] = caculate_metric_output["APY"]
            final_profit_result.loc[2, MODEL_NAME] = caculate_metric_output["SR"]
        if economic_distiller_caculate_metric_output != None:
            daily_cumulative_wealth[MODEL_NAME + ' (ED)'] = economic_distiller_caculate_metric_output["DCW"]
            final_profit_result = final_profit_result.assign(**{MODEL_NAME + ' (ED)': 0})
            final_profit_result.loc[0, MODEL_NAME + ' (ED)'] = economic_distiller_caculate_metric_output["CW"]
            final_profit_result.loc[1, MODEL_NAME + ' (ED)'] = economic_distiller_caculate_metric_output["APY"]
            final_profit_result.loc[2, MODEL_NAME + ' (ED)'] = economic_distiller_caculate_metric_output["SR"]
        print(
            final_profit_result
        )
        plot_dataframe(daily_cumulative_wealth, PLOT_ALL_1, 'DCW', logdir)
        # plot_dataframe(daily_cumulative_wealth, PLOT_ALL_2, 'DCW', logdir)
        # plot_dataframe(daily_cumulative_wealth, PLOT_ALL_3, 'DCW', logdir)
        # plot_dataframe(daily_cumulative_wealth, PLOT_ALL_4, 'DCW', logdir)
        # plot_dataframe(daily_cumulative_wealth, PLOT_ALL_5, 'DCW', logdir)

    if METRIC_CONFIG.get('INCLUDE_RISK_METRICS'):
        daily_drawdown = pd.read_excel(ROOT_PATH + '/data/benchmark_results/risk_metrics/'+DATASET_NAME+'/daily_drawdown.xlsx')
        final_risk_result = pd.read_excel(ROOT_PATH + '/data/benchmark_results/risk_metrics/'+DATASET_NAME+'/final_risk_result.xlsx')

        daily_drawdown = daily_drawdown.dropna(axis=1, how="any")
        final_risk_result = final_risk_result.dropna(axis=1, how="any")
        if caculate_metric_output != None:
            daily_drawdown[MODEL_NAME] = caculate_metric_output["DDD"]
            final_risk_result = final_risk_result.assign(**{MODEL_NAME: 0})
            final_risk_result.loc[0, MODEL_NAME] = caculate_metric_output["VR"]
            final_risk_result.loc[1, MODEL_NAME] = caculate_metric_output["MDD"]
        if economic_distiller_caculate_metric_output != None:
            daily_drawdown[MODEL_NAME + ' (ED)'] = economic_distiller_caculate_metric_output["DDD"]
            final_risk_result = final_risk_result.assign(**{MODEL_NAME + ' (ED)': 0})
            final_risk_result.loc[0, MODEL_NAME + ' (ED)'] = economic_distiller_caculate_metric_output["VR"]
            final_risk_result.loc[1, MODEL_NAME + ' (ED)'] = economic_distiller_caculate_metric_output["MDD"]
        print(
            final_risk_result
        )
        plot_dataframe(daily_drawdown, PLOT_ALL_1, 'DDD', logdir)
        # plot_dataframe(daily_drawdown, PLOT_ALL_2, 'DDD', logdir)
        # plot_dataframe(daily_drawdown, PLOT_ALL_3, 'DDD', logdir)
        # plot_dataframe(daily_drawdown, PLOT_ALL_4, 'DDD', logdir)
        # plot_dataframe(daily_drawdown, PLOT_ALL_5, 'DDD', logdir)

    if METRIC_CONFIG.get('PRACTICAL_METRICS')['INCLUDE_PRACTICAL_METRICS']:
        transaction_costs_adjusted_cumulative_wealth = pd.read_excel(ROOT_PATH + '/data/benchmark_results/practical_metrics/'+DATASET_NAME+'/transaction_costs_adjusted_cumulative_wealth.xlsx')
        final_practical_result = pd.read_excel(ROOT_PATH + '/data/benchmark_results/practical_metrics/'+DATASET_NAME+'/final_practical_result.xlsx')

        transaction_costs_adjusted_cumulative_wealth = transaction_costs_adjusted_cumulative_wealth.dropna(axis=1, how="any")
        final_practical_result = final_practical_result.dropna(axis=1, how="any")
        if caculate_metric_output != None:
            transaction_costs_adjusted_cumulative_wealth[MODEL_NAME] = caculate_metric_output["TCW"]
            final_practical_result = final_practical_result.assign(**{MODEL_NAME: 0})
            final_practical_result.loc[0, MODEL_NAME] = caculate_metric_output["ATO"]
            final_practical_result.loc[1, MODEL_NAME] = caculate_metric_output["RT"]
        if economic_distiller_caculate_metric_output != None:
            transaction_costs_adjusted_cumulative_wealth[MODEL_NAME + ' (ED)'] = economic_distiller_caculate_metric_output["TCW"]
            final_practical_result = final_practical_result.assign(**{MODEL_NAME + ' (ED)': 0})
            final_practical_result.loc[0, MODEL_NAME + ' (ED)'] = economic_distiller_caculate_metric_output["ATO"]
            final_practical_result.loc[1, MODEL_NAME + ' (ED)'] = economic_distiller_caculate_metric_output["RT"]
        print(
            final_practical_result
        )
        plot_dataframe(transaction_costs_adjusted_cumulative_wealth, PLOT_ALL_1, 'TCW', logdir)
        # plot_dataframe(transaction_costs_adjusted_cumulative_wealth, PLOT_ALL_2, 'TCW', logdir)
        # plot_dataframe(transaction_costs_adjusted_cumulative_wealth, PLOT_ALL_3, 'TCW', logdir)
        # plot_dataframe(transaction_costs_adjusted_cumulative_wealth, PLOT_ALL_4, 'TCW', logdir)
        # plot_dataframe(transaction_costs_adjusted_cumulative_wealth, PLOT_ALL_5, 'TCW', logdir)

    load_benchmark_output = {}
    load_benchmark_output['logdir'] = caculate_metric_output['logdir']
    load_benchmark_output['CW'] = caculate_metric_output["CW"]
    load_benchmark_output['TCW'] = caculate_metric_output["TCW"].iloc[-6]
    print(load_benchmark_output['TCW'])
    return load_benchmark_output

if __name__ == '__main__':
    # utils.check_update()
    load_benchmark()