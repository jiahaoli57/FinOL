import warnings
import pandas as pd
import matplotlib.pyplot as plt

from finol.evaluation_layer.radar_chart import plot_radar_chart
from finol.utils import ROOT_PATH, load_config

plt.style.use("seaborn-v0_8-paper")
plt.rcParams["font.family"] = "Microsoft YaHei"
warnings.filterwarnings("ignore")
warnings.simplefilter(action="ignore", category=FutureWarning)


class BenchmarkLoader:
    def __init__(self, caculate_metric_output, economic_distiller_caculate_metric_output):
        self.config = load_config()
        self.caculate_metric_output = caculate_metric_output
        self.economic_distiller_caculate_metric_output = economic_distiller_caculate_metric_output
        self.logdir = self.caculate_metric_output["logdir"]

    def plot_dataframe(self, df, plot_type):
        if plot_type == "DCW":
            df_copy = df.copy()
            df_copy = df_copy.drop(df.columns[-1], axis=1)
            df_copy.set_index('DATE', inplace=True)
            last_row = df_copy.iloc[-1]
            max_columns = last_row.nlargest(5).index.tolist()
            column_names = max_columns + [self.config["MODEL_NAME"]]
            print(
                column_names
            )
        else:
            column_names = self.config["COMPARED_BASELINE"] + [self.config["MODEL_NAME"]]

        if self.config["INTERPRETABLE_ANALYSIS_CONFIG"]["INCLUDE_ECONOMIC_DISTILLATION"]:
            column_names = column_names + [self.config["MODEL_NAME"] + ' (ED)']

        fig = plt.figure()  # figsize=(9, 4)

        if plot_type == "DCW":
            markevery = self.config["MARKEVERY"][self.config["DATASET_NAME"]]
            if self.config["PLOT_CHINESE"]:
                xlabel = "交易期"
                ylabel = "逐期累积财富"
            else:
                xlabel = "Trading Periods"
                ylabel = "Daily Cumulative Wealth"

        elif plot_type == "DDD":
            markevery = self.config["MARKEVERY"][self.config["DATASET_NAME"]]
            if self.config["PLOT_CHINESE"]:
                xlabel = "交易期"
                ylabel = "逐期下行风险"
            else:
                xlabel = "Trading Periods"
                ylabel = "Daily DrawDown"

        elif plot_type == "TCW":
            plt.figure()  # figsize=(6, 5)
            # TRANSACTIOS_COSTS_RATE = self.config["METRIC_CONFIG"].get("PRACTICAL_METRICS")["TRANSACTIOS_COSTS_RATE"]
            # TRANSACTIOS_COSTS_RATE_INTERVAL = self.config["METRIC_CONFIG"].get("PRACTICAL_METRICS")["TRANSACTIOS_COSTS_RATE_INTERVAL"]
            TRANSACTIOS_COSTS_RATE = self.config["METRIC_CONFIG"]["PRACTICAL_METRICS"]["TRANSACTIOS_COSTS_RATE"]
            TRANSACTIOS_COSTS_RATE_INTERVAL = self.config["METRIC_CONFIG"]["PRACTICAL_METRICS"]["TRANSACTIOS_COSTS_RATE_INTERVAL"]
            markevery = 1
            if self.config["PLOT_CHINESE"]:
                xlabel = "交易费用率 (%)"
                ylabel = "考虑交易费用的累积财富"
            else:
                xlabel = "Transaction Costs Rates (%)"
                ylabel = "Costs-Adjusted Cumulative Wealth"

            plt.xticks(ticks=[0, 2, 4, 6, 8, 10], labels=[0, 0.2, 0.4, 0.6, 0.8, 1])
            plt.xticks(ticks=[0, 2.5, 5, 7.5, 10], labels=[0, 0.25, 0.5, 0.75, 1])
            plt.xticks(ticks=[0, 1.25, 2.5, 3.75, 5, 7.5, 10], labels=[0, 0.125, 0.25, 0.375, 0.5, 0.75, 1])

        df = df[column_names]
        # df = df.rename(columns={df.columns[-2]: "Teacher-Model", df.columns[-1]: "Student-Model"})
        num_columns = len(df.columns)

        if self.config["INTERPRETABLE_ANALYSIS_CONFIG"]["INCLUDE_ECONOMIC_DISTILLATION"]:
            colors = ["black"] * (num_columns - 2) + ["red"] * 2
            lines = ["-"] * (num_columns - 1) + [":"]
        else:
            colors = ["black"] * (num_columns - 1) + ["red"] * 1
            lines = ["-"] * (num_columns)

        if plot_type == "TCW":
            for i, column in enumerate(df.columns):
                plt.plot(df[column].head(int(TRANSACTIOS_COSTS_RATE/TRANSACTIOS_COSTS_RATE_INTERVAL)+1), linestyle=lines[i], color=colors[i], marker=self.config["MARKERS"][i], markevery=markevery, alpha=0.5,
                         label=column)
        else:
            for i, column in enumerate(df.columns):
                plt.plot(df[column], linestyle=lines[i], color=colors[i], marker=self.config["MARKERS"][i], markevery=markevery, alpha=0.5,
                         label=column)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(self.config["DATASET_NAME"])
        plt.legend(loc="best")  # lower left  upper left
        plt.grid(True)
        # plt.yscale("log")
        plt.tight_layout()
        plt.savefig(self.logdir + "/" + self.config["MODEL_NAME"] + "_" + self.config["DATASET_NAME"] + "_" + plot_type + ".pdf",
                    format="pdf",
                    dpi=300,
                    bbox_inches="tight")
        plt.show()

    def load_benchmark(self):
        """
        Load baseline results from a CSV file for specified target classes.

        Returns:
        - None
        """
        if self.caculate_metric_output != None:
            logdir = self.caculate_metric_output["logdir"]
        else:
            logdir = ROOT_PATH

        # if METRIC_CONFIG.get("INCLUDE_PROFIT_METRICS"):
        daily_return = pd.read_excel(ROOT_PATH + "/data/benchmark_results/profit_metrics/" + self.config["DATASET_NAME"] + "/daily_return.xlsx")
        daily_cumulative_wealth = pd.read_excel(ROOT_PATH + "/data/benchmark_results/profit_metrics/" + self.config["DATASET_NAME"] + "/daily_cumulative_wealth.xlsx")
        final_profit_result = pd.read_excel(ROOT_PATH + "/data/benchmark_results/profit_metrics/" + self.config["DATASET_NAME"] + "/final_profit_result.xlsx")

        daily_return = daily_return.dropna(axis=1, how="any")
        daily_cumulative_wealth = daily_cumulative_wealth.dropna(axis=1, how="any")
        final_profit_result = final_profit_result.dropna(axis=1, how="any")
        if self.caculate_metric_output != None:
            daily_cumulative_wealth[self.config["MODEL_NAME"]] = self.caculate_metric_output["DCW"]
            final_profit_result = final_profit_result.assign(**{self.config["MODEL_NAME"]: 0})
            final_profit_result.loc[0, self.config["MODEL_NAME"]] = self.caculate_metric_output["CW"]
            final_profit_result.loc[1, self.config["MODEL_NAME"]] = self.caculate_metric_output["APY"]
            final_profit_result.loc[2, self.config["MODEL_NAME"]] = self.caculate_metric_output["SR"]
        if self.economic_distiller_caculate_metric_output != None:
            daily_cumulative_wealth[self.config["MODEL_NAME"] + " (ED)"] = self.economic_distiller_caculate_metric_output["DCW"]
            final_profit_result = final_profit_result.assign(**{self.config["MODEL_NAME"] + " (ED)": 0})
            final_profit_result.loc[0, self.config["MODEL_NAME"] + " (ED)"] = self.economic_distiller_caculate_metric_output["CW"]
            final_profit_result.loc[1, self.config["MODEL_NAME"] + " (ED)"] = self.economic_distiller_caculate_metric_output["APY"]
            final_profit_result.loc[2, self.config["MODEL_NAME"] + " (ED)"] = self.economic_distiller_caculate_metric_output["SR"]
        print(
            final_profit_result
        )
        # dd = final_profit_result.drop("Metric", axis=1).T
        # dd.columns = ["CW", "APY", "SR"]
        # dd = dd.reset_index(drop=False)
        # print(
        #     dd
        # )
        # # time.sleep(1111)
        # import dash
        # from dash import dash_table
        # from dash import html
        # # Create Dash app
        # app = dash.Dash(__name__)
        # # Create Dash table from dataframe
        # table = dash_table.DataTable(
        #   data=dd.to_dict("records"),
        #   columns=[{"name": i, "id": i} for i in dd.columns]
        # )
        # # Add sorting
        # table.sort_action = "native"
        # app.layout = html.Div([
        #     html.H1("Interactive DataTable"),
        #
        #     html.Div([
        #         html.H2("DataFrame 1"),
        #         table
        #     ]),
        #
        #     html.Div([
        #         html.H3("DataFrame 2"),
        #         table
        #     ]),
        # ])
        # app.run_server(debug=True)
        # time.sleep(1111)

        self.plot_dataframe(daily_cumulative_wealth, "DCW")
        # plot_dataframe(daily_cumulative_wealth, PLOT_ALL_2, "DCW", logdir)
        # plot_dataframe(daily_cumulative_wealth, PLOT_ALL_3, "DCW", logdir)
        # plot_dataframe(daily_cumulative_wealth, PLOT_ALL_4, "DCW", logdir)
        # plot_dataframe(daily_cumulative_wealth, PLOT_ALL_5, "DCW", logdir)

        # if METRIC_CONFIG.get("INCLUDE_RISK_METRICS"):
        daily_drawdown = pd.read_excel(ROOT_PATH + "/data/benchmark_results/risk_metrics/" + self.config["DATASET_NAME"] + "/daily_drawdown.xlsx")
        final_risk_result = pd.read_excel(ROOT_PATH + "/data/benchmark_results/risk_metrics/" + self.config["DATASET_NAME"] + "/final_risk_result.xlsx")

        daily_drawdown = daily_drawdown.dropna(axis=1, how="any")
        final_risk_result = final_risk_result.dropna(axis=1, how="any")
        if self.caculate_metric_output != None:
            daily_drawdown[self.config["MODEL_NAME"]] = self.caculate_metric_output["DDD"]
            final_risk_result = final_risk_result.assign(**{self.config["MODEL_NAME"]: 0})
            final_risk_result.loc[0, self.config["MODEL_NAME"]] = self.caculate_metric_output["VR"]
            final_risk_result.loc[1, self.config["MODEL_NAME"]] = self.caculate_metric_output["MDD"]
        if self.economic_distiller_caculate_metric_output != None:
            daily_drawdown[self.config["MODEL_NAME"] + " (ED)"] = self.economic_distiller_caculate_metric_output["DDD"]
            final_risk_result = final_risk_result.assign(**{self.config["MODEL_NAME"] + " (ED)": 0})
            final_risk_result.loc[0, self.config["MODEL_NAME"] + " (ED)"] = self.economic_distiller_caculate_metric_output["VR"]
            final_risk_result.loc[1, self.config["MODEL_NAME"] + " (ED)"] = self.economic_distiller_caculate_metric_output["MDD"]
        print(
            final_risk_result
        )
        self.plot_dataframe(daily_drawdown, "DDD")
        # plot_dataframe(daily_drawdown, PLOT_ALL_2, "DDD", logdir)
        # plot_dataframe(daily_drawdown, PLOT_ALL_3, "DDD", logdir)
        # plot_dataframe(daily_drawdown, PLOT_ALL_4, "DDD", logdir)
        # plot_dataframe(daily_drawdown, PLOT_ALL_5, "DDD", logdir)

        # if METRIC_CONFIG.get("PRACTICAL_METRICS")["INCLUDE_PRACTICAL_METRICS"]:
        transaction_costs_adjusted_cumulative_wealth = pd.read_excel(ROOT_PATH + "/data/benchmark_results/practical_metrics/" + self.config["DATASET_NAME"] + "/transaction_costs_adjusted_cumulative_wealth.xlsx")
        final_practical_result = pd.read_excel(ROOT_PATH + "/data/benchmark_results/practical_metrics/" + self.config["DATASET_NAME"] + "/final_practical_result.xlsx")

        transaction_costs_adjusted_cumulative_wealth = transaction_costs_adjusted_cumulative_wealth.dropna(axis=1, how="any")
        final_practical_result = final_practical_result.dropna(axis=1, how="any")
        if self.caculate_metric_output != None:
            transaction_costs_adjusted_cumulative_wealth[self.config["MODEL_NAME"]] = self.caculate_metric_output["TCW"]
            final_practical_result = final_practical_result.assign(**{self.config["MODEL_NAME"]: 0})
            final_practical_result.loc[0, self.config["MODEL_NAME"]] = self.caculate_metric_output["ATO"]
            final_practical_result.loc[1, self.config["MODEL_NAME"]] = self.caculate_metric_output["RT"]
        if self.economic_distiller_caculate_metric_output != None:
            transaction_costs_adjusted_cumulative_wealth[self.config["MODEL_NAME"] + " (ED)"] = self.economic_distiller_caculate_metric_output["TCW"]
            final_practical_result = final_practical_result.assign(**{self.config["MODEL_NAME"] + " (ED)": 0})
            final_practical_result.loc[0, self.config["MODEL_NAME"] + " (ED)"] = self.economic_distiller_caculate_metric_output["ATO"]
            final_practical_result.loc[1, self.config["MODEL_NAME"] + " (ED)"] = self.economic_distiller_caculate_metric_output["RT"]
        print(
            final_practical_result
        )
        self.plot_dataframe(transaction_costs_adjusted_cumulative_wealth, "TCW")
        # plot_dataframe(transaction_costs_adjusted_cumulative_wealth, PLOT_ALL_2, "TCW", logdir)
        # plot_dataframe(transaction_costs_adjusted_cumulative_wealth, PLOT_ALL_3, "TCW", logdir)
        # plot_dataframe(transaction_costs_adjusted_cumulative_wealth, PLOT_ALL_4, "TCW", logdir)
        # plot_dataframe(transaction_costs_adjusted_cumulative_wealth, PLOT_ALL_5, "TCW", logdir)

        plot_radar_chart(final_profit_result, final_risk_result, self.COMPARED_BASELINE, logdir)

        load_benchmark_output = {}
        load_benchmark_output["logdir"] = self.caculate_metric_output["logdir"]
        load_benchmark_output["CW"] = self.caculate_metric_output["CW"]
        load_benchmark_output["TCW"] = self.caculate_metric_output["TCW"].iloc[-6]
        print(load_benchmark_output["TCW"])
        return load_benchmark_output


# if __name__ == "__main__":
