import time
import warnings
import pandas as pd
import numpy as np


from tabulate import tabulate
from typing import List, Tuple, Dict, Union, Any
# from finol.evaluation_layer.radar_chart import plot_radar_chart
from finol.utils import ROOT_PATH, load_config, add_prefix

# plt.style.use("seaborn-paper")
# plt.rcParams["font.family"] = "Microsoft YaHei"
# warnings.filterwarnings("ignore")
# warnings.simplefilter(action="ignore", category=FutureWarning)


class BenchmarkLoader:
    """
    Class to load the benchmarks and perform comparisons.

    :param calculate_metric_output: Dictionary containing output from function :func:`~finol.evaluation_layer.MetricCalculator.calculate_metric`.
    :param economic_distillation_output: Dictionary containing output from function :func:`~finol.evaluation_layer.EconomicDistiller.economic_distillation`.
    """
    def __init__(self, calculate_metric_output: Dict, economic_distillation_output: Dict) -> None:
        self.config = load_config()
        self.calculate_metric_output = calculate_metric_output
        self.economic_distillation_output = economic_distillation_output
        self.logdir = self.calculate_metric_output["logdir"]

    def find_top_5_baselines(self, df: pd.DataFrame) -> None:
        """
        Find the top 5 baselines based on the provided Benchmark DataFrame.

        :param df: Benchmark DataFrame containing all baselines.
        """
        df_copy = df.copy()
        df_copy = df_copy.drop(df.columns[-1], axis=1)
        df_copy.set_index('DATE', inplace=True)
        last_row = df_copy.iloc[-1]
        max_columns = last_row.nlargest(5).index.tolist()
        self.top_5_baselines = max_columns + [self.config["MODEL_NAME"]]
        if self.config["INTERPRETABLE_ANALYSIS_CONFIG"]["INCLUDE_ECONOMIC_DISTILLATION"]:
            self.top_5_baselines = self.top_5_baselines + [self.config["MODEL_NAME"] + ' (ED)']

    def load_benchmark(self) -> Dict[str, Any]:
        """
        Load benchmark data and update final profit results with model metrics.

        :return: Dictionary containing benchmark results and model's results.
        """
        logdir = self.calculate_metric_output["logdir"]

        # ------------------------------------------------------------------------------------------------------------ #
        daily_return = pd.read_excel(ROOT_PATH + "/data/benchmark_results/profit_metrics/" + self.config["DATASET_NAME"] + "/daily_return.xlsx")
        daily_cumulative_wealth = pd.read_excel(ROOT_PATH + "/data/benchmark_results/profit_metrics/" + self.config["DATASET_NAME"] + "/daily_cumulative_wealth.xlsx")
        final_profit_result = pd.read_excel(ROOT_PATH + "/data/benchmark_results/profit_metrics/" + self.config["DATASET_NAME"] + "/final_profit_result.xlsx")

        daily_return = daily_return.dropna(axis=1, how="any")
        daily_cumulative_wealth = daily_cumulative_wealth.dropna(axis=1, how="any")
        final_profit_result = final_profit_result.dropna(axis=1, how="any")
        if self.calculate_metric_output is not None:
            daily_cumulative_wealth[self.config["MODEL_NAME"]] = self.calculate_metric_output["DCW"]
            final_profit_result = final_profit_result.assign(**{self.config["MODEL_NAME"]: np.nan})
            final_profit_result.loc[0, self.config["MODEL_NAME"]] = self.calculate_metric_output["CW"]
            final_profit_result.loc[1, self.config["MODEL_NAME"]] = self.calculate_metric_output["APY"]
            final_profit_result.loc[2, self.config["MODEL_NAME"]] = self.calculate_metric_output["SR"]
        if self.economic_distillation_output is not None:
            daily_cumulative_wealth[self.config["MODEL_NAME"] + " (ED)"] = self.economic_distillation_output["DCW"]
            final_profit_result = final_profit_result.assign(**{self.config["MODEL_NAME"] + " (ED)": np.nan})
            final_profit_result.loc[0, self.config["MODEL_NAME"] + " (ED)"] = self.economic_distillation_output["CW"]
            final_profit_result.loc[1, self.config["MODEL_NAME"] + " (ED)"] = self.economic_distillation_output["APY"]
            final_profit_result.loc[2, self.config["MODEL_NAME"] + " (ED)"] = self.economic_distillation_output["SR"]

        self.find_top_5_baselines(daily_cumulative_wealth)

        tabulate_data = []
        tabulate_data.append(["FCW"] + list(final_profit_result.loc[0, self.top_5_baselines].values))
        tabulate_data.append(["APY"] + list(final_profit_result.loc[1, self.top_5_baselines].values))
        tabulate_data.append(["SR"] + list(final_profit_result.loc[2, self.top_5_baselines].values))
        print("Profitability comparison with the top five baselines:")
        print(tabulate(tabulate_data, headers=["Profit Metric"] + list(final_profit_result.loc[0, self.top_5_baselines].index), tablefmt="psql", numalign="left"))

        # save results
        daily_cumulative_wealth.to_excel(logdir + "/" + add_prefix("daily_cumulative_wealth.xlsx"), index=False)
        final_profit_result.to_excel(logdir + "/" + add_prefix("final_profit_result.xlsx"), index=False)

        # ------------------------------------------------------------------------------------------------------------ #
        daily_drawdown = pd.read_excel(ROOT_PATH + "/data/benchmark_results/risk_metrics/" + self.config["DATASET_NAME"] + "/daily_drawdown.xlsx")
        final_risk_result = pd.read_excel(ROOT_PATH + "/data/benchmark_results/risk_metrics/" + self.config["DATASET_NAME"] + "/final_risk_result.xlsx")

        daily_drawdown = daily_drawdown.dropna(axis=1, how="any")
        final_risk_result = final_risk_result.dropna(axis=1, how="any")

        # calculate daily MDD
        daily_maximum_drawdown = daily_drawdown.copy()
        for col in daily_drawdown.columns:
            daily_maximum_drawdown[f"{col}"] = daily_drawdown[col].cummax()

        if self.calculate_metric_output != None:
            daily_drawdown[self.config["MODEL_NAME"]] = self.calculate_metric_output["DDD"]
            daily_maximum_drawdown[self.config["MODEL_NAME"]] = self.calculate_metric_output["DMDD"]
            final_risk_result = final_risk_result.assign(**{self.config["MODEL_NAME"]: np.nan})
            final_risk_result.loc[0, self.config["MODEL_NAME"]] = self.calculate_metric_output["VR"]
            final_risk_result.loc[1, self.config["MODEL_NAME"]] = self.calculate_metric_output["MDD"]
        if self.economic_distillation_output != None:
            daily_drawdown[self.config["MODEL_NAME"] + " (ED)"] = self.economic_distillation_output["DDD"]
            daily_maximum_drawdown[self.config["MODEL_NAME"] + " (ED)"] = self.economic_distillation_output["DMDD"]
            final_risk_result = final_risk_result.assign(**{self.config["MODEL_NAME"] + " (ED)": np.nan})
            final_risk_result.loc[0, self.config["MODEL_NAME"] + " (ED)"] = self.economic_distillation_output["VR"]
            final_risk_result.loc[1, self.config["MODEL_NAME"] + " (ED)"] = self.economic_distillation_output["MDD"]

        tabulate_data = []
        tabulate_data.append(["VR"] + list(final_risk_result.loc[0, self.top_5_baselines].values))
        tabulate_data.append(["MDD"] + list(final_risk_result.loc[1, self.top_5_baselines].values))
        print("Risk resilience comparison with the top five baselines:")
        print(tabulate(tabulate_data, headers=["Risk Metric"] + list(final_risk_result.loc[0, self.top_5_baselines].index), tablefmt="psql", numalign="left"))

        # save results
        daily_drawdown.to_excel(logdir + "/" + add_prefix("daily_drawdown.xlsx"), index=False)
        daily_maximum_drawdown.to_excel(logdir + "/" + add_prefix("daily_maximumdrawdown.xlsx"), index=False)
        final_risk_result.to_excel(logdir + "/" + add_prefix("final_risk_result.xlsx"), index=False)

        # ------------------------------------------------------------------------------------------------------------ #
        transaction_costs_adjusted_cumulative_wealth = pd.read_excel(ROOT_PATH + "/data/benchmark_results/practical_metrics/" + self.config["DATASET_NAME"] + "/transaction_costs_adjusted_cumulative_wealth.xlsx")
        final_practical_result = pd.read_excel(ROOT_PATH + "/data/benchmark_results/practical_metrics/" + self.config["DATASET_NAME"] + "/final_practical_result.xlsx")

        transaction_costs_adjusted_cumulative_wealth = transaction_costs_adjusted_cumulative_wealth.dropna(axis=1, how="any")
        final_practical_result = final_practical_result.dropna(axis=1, how="any")
        if self.calculate_metric_output != None:
            transaction_costs_adjusted_cumulative_wealth[self.config["MODEL_NAME"]] = self.calculate_metric_output["TCW"]
            final_practical_result = final_practical_result.assign(**{self.config["MODEL_NAME"]: np.nan})
            final_practical_result.loc[0, self.config["MODEL_NAME"]] = self.calculate_metric_output["ATO"]
            final_practical_result.loc[1, self.config["MODEL_NAME"]] = self.calculate_metric_output["RT"]
        if self.economic_distillation_output != None:
            transaction_costs_adjusted_cumulative_wealth[self.config["MODEL_NAME"] + " (ED)"] = self.economic_distillation_output["TCW"]
            final_practical_result = final_practical_result.assign(**{self.config["MODEL_NAME"] + " (ED)": np.nan})
            final_practical_result.loc[0, self.config["MODEL_NAME"] + " (ED)"] = self.economic_distillation_output["ATO"]
            final_practical_result.loc[1, self.config["MODEL_NAME"] + " (ED)"] = self.economic_distillation_output["RT"]

        tabulate_data = []
        tabulate_data.append(["ATO"] + list(final_practical_result.loc[0, self.top_5_baselines].values))
        tabulate_data.append(["RT"] + list(final_practical_result.loc[1, self.top_5_baselines].values))
        print("Practicality comparison with the top five baselines:")
        print(tabulate(tabulate_data, headers=["Practical Metric"] + list(final_practical_result.loc[0, self.top_5_baselines].index), tablefmt="psql", numalign="left"))

        # save results
        transaction_costs_adjusted_cumulative_wealth.to_excel(logdir + "/" + add_prefix("transaction_costs_adjusted_cumulative_wealth.xlsx"), index=False)
        final_practical_result.to_excel(logdir + "/" + add_prefix("final_practical_result.xlsx"), index=False)

        # ------------------------------------------------------------------------------------------------------------ #
        # plot_radar_chart(final_profit_result, final_risk_result, self.COMPARED_BASELINE, logdir)

        load_benchmark_output = {}
        load_benchmark_output["logdir"] = self.calculate_metric_output["logdir"]
        load_benchmark_output["top_5_baselines"] = self.top_5_baselines
        load_benchmark_output["daily_cumulative_wealth"] = daily_cumulative_wealth
        load_benchmark_output["final_profit_result"] = final_profit_result
        load_benchmark_output["daily_drawdown"] = daily_drawdown
        load_benchmark_output["daily_maximum_drawdown"] = daily_maximum_drawdown
        load_benchmark_output["final_risk_result"] = final_risk_result
        load_benchmark_output["transaction_costs_adjusted_cumulative_wealth"] = transaction_costs_adjusted_cumulative_wealth
        load_benchmark_output["final_practical_result"] = final_practical_result
        # load_benchmark_output["CW"] = self.calculate_metric_output["CW"]
        # load_benchmark_output["TCW"] = self.calculate_metric_output["TCW"]
        # print(load_benchmark_output["TCW"])
        return load_benchmark_output


# if __name__ == "__main__":
