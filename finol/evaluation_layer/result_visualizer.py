import matplotlib.pyplot as plt

from finol.utils import load_config, add_prefix

plt.rcParams["font.family"] = "Microsoft YaHei"
class ResultVisualizer:
    def __init__(self, load_benchmark_output) -> None:
        self.config = load_config()
        self.load_benchmark_output = load_benchmark_output
        self.top_5_baselines = load_benchmark_output["top_5_baselines"]
        self.logdir = load_benchmark_output["logdir"]

    def visualize_result(self) -> None:
        """
        Visualize the result using appropriate plotting methods.

        :return: None
        """
        # ["DCW", "DDD", "DMDD", "TCW"]
        for plot_type in ["DCW", "DMDD", "TCW"]:
            self.plot_type = plot_type
            if plot_type == "DCW":
                self._plot_daily_cumulative_wealth()
            elif plot_type == "DMDD":
                self._plot_daily_maximum_drawdown()
            elif plot_type == "TCW":
                self._plot_transaction_costs_adjusted_wealth()

    def _plot_daily_cumulative_wealth(self) -> None:
        """
        Plot the daily cumulative wealth for the top baselines.

        This method plots the daily cumulative wealth for the top baselines based on the configuration.
        It retrieves the necessary data, sets labels based on the language configuration, plots the data with
        appropriate styles, and saves the plot as a PDF file.

        :return: None
        """
        df = self.load_benchmark_output["daily_cumulative_wealth"]
        df = df[self.top_5_baselines]
        num_columns = len(df.columns)

        plot_labels = {
            "en": ("Trading Periods", "Daily Cumulative Wealth"),
            "zh-CHS": ("交易期", "逐期累积财富"),
            "zh-CHT": ("交易期", "逐期累積財富")
        }
        xlabel, ylabel = plot_labels[self.config["PLOT_LANGUAGE"]]

        lines = ["-"] * (num_columns)
        colors = ["black"] * (num_columns - 1) + ["red"] * 1

        for i, column in enumerate(df.columns):
            plt.plot(
                df[column],
                linestyle=lines[i],
                color=colors[i],
                marker=self.config["MARKERS"][i],
                markevery=self.config["MARKEVERY"][self.config["DATASET_NAME"]],
                alpha=0.5,
                label=column
            )

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(self.config["DATASET_NAME"])
        plt.legend(loc="best")  # lower left upper left
        plt.grid(True)
        # plt.yscale("log")
        plt.tight_layout()
        plt.savefig(self.logdir + "/" + add_prefix(self.plot_type) + ".pdf", format="pdf", dpi=300, bbox_inches="tight")
        plt.show()
        plt.figure()


    def _plot_daily_drawdown(self) -> None:
        """"
        :return: None
        """
        df = self.load_benchmark_output["daily_daily_drawdown"]
        df = df[self.top_5_baselines]
        num_columns = len(df.columns)

        plot_labels = {
            "en": ("Trading Periods", "Daily DrawDown"),
            "zh-CHS": ("交易期", "逐期下行风险"),
            "zh-CHT": ("交易期", "逐期下行風險"),
        }
        xlabel, ylabel = plot_labels[self.config["PLOT_LANGUAGE"]]

        if self.config["INTERPRETABLE_ANALYSIS_CONFIG"]["INCLUDE_ECONOMIC_DISTILLATION"]:
            lines = ["-"] * (num_columns - 1) + [":"]
            colors = ["black"] * (num_columns - 2) + ["red"] * 2
        else:
            lines = ["-"] * (num_columns)
            colors = ["black"] * (num_columns - 1) + ["red"] * 1

        for i, column in enumerate(df.columns):
            plt.plot(
                df[column],
                linestyle=lines[i],
                color=colors[i],
                marker=self.config["MARKERS"][i],
                markevery=self.config["MARKEVERY"][self.config["DATASET_NAME"]],
                alpha=0.5,
                label=column
            )

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(self.config["DATASET_NAME"])
        plt.legend(loc="best")  # lower left upper left
        plt.grid(True)
        # plt.yscale("log")
        plt.tight_layout()
        plt.savefig(self.logdir + "/" + add_prefix(self.plot_type) + ".pdf", format="pdf", dpi=300, bbox_inches="tight")
        plt.show()
        plt.figure()


    def _plot_daily_maximum_drawdown(self) -> None:
        """

        :return: None
        """
        df = self.load_benchmark_output["daily_maximum_drawdown"]
        df = df[self.top_5_baselines]
        num_columns = len(df.columns)

        plot_labels = {
            "en": ("Trading Periods", "Daily Maximum DrawDown"),
            "zh-CHS": ("交易期", "逐期最大下行风险"),
            "zh-CHT": ("交易期", "逐期最大下行風險"),
        }
        xlabel, ylabel = plot_labels[self.config["PLOT_LANGUAGE"]]

        if self.config["INTERPRETABLE_ANALYSIS_CONFIG"]["INCLUDE_ECONOMIC_DISTILLATION"]:
            lines = ["-"] * (num_columns - 1) + [":"]
            colors = ["black"] * (num_columns - 2) + ["red"] * 2
        else:
            lines = ["-"] * (num_columns)
            colors = ["black"] * (num_columns - 1) + ["red"] * 1

        for i, column in enumerate(df.columns):
            plt.plot(
                df[column],
                linestyle=lines[i],
                color=colors[i],
                marker=self.config["MARKERS"][i],
                markevery=self.config["MARKEVERY"][self.config["DATASET_NAME"]],
                alpha=0.5,
                label=column
            )

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(self.config["DATASET_NAME"])
        plt.legend(loc="best")  # lower left upper left
        plt.grid(True)
        # plt.yscale("log")
        plt.tight_layout()
        plt.savefig(self.logdir + "/" + add_prefix(self.plot_type) + ".pdf", format="pdf", dpi=300, bbox_inches="tight")
        plt.show()
        plt.figure()

    def _plot_transaction_costs_adjusted_wealth(self) -> None:
        """

        :return: None
        """

        df = self.load_benchmark_output["transaction_costs_adjusted_cumulative_wealth"]
        df = df[self.top_5_baselines]
        num_columns = len(df.columns)

        plot_labels = {
            "en": ("Transaction Costs Rates (%)", "Costs-Adjusted Cumulative Wealth"),
            "zh-CHS": ("交易费用率( %)", "考虑交易费用的累积财富"),
            "zh-CHT": ("交易費用率( %)", "考慮交易費用的累積財富"),
        }
        xlabel, ylabel = plot_labels[self.config["PLOT_LANGUAGE"]]

        if self.config["INTERPRETABLE_ANALYSIS_CONFIG"]["INCLUDE_ECONOMIC_DISTILLATION"]:
            lines = ["-"] * (num_columns - 1) + [":"]
            colors = ["black"] * (num_columns - 2) + ["red"] * 2
        else:
            lines = ["-"] * (num_columns)
            colors = ["black"] * (num_columns - 1) + ["red"] * 1

        # plt.xticks(ticks=[0, 2, 4, 6, 8, 10], labels=[0, 0.2, 0.4, 0.6, 0.8, 1])
        # plt.xticks(ticks=[0, 2.5, 5, 7.5, 10], labels=[0, 0.25, 0.5, 0.75, 1])
        plt.xticks(ticks=[0, 1.25, 2.5, 3.75, 5, 7.5, 10], labels=[0, 0.125, 0.25, 0.375, 0.5, 0.75, 1])

        # tc = 0.005
        # interval = 0.001
        for i, column in enumerate(df.columns):
            plt.plot(
                df[column].head(int(0.005/0.001)+1),
                linestyle=lines[i],
                color=colors[i],
                marker=self.config["MARKERS"][i],
                markevery=1,
                alpha=0.5,
                label=column
            )

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(self.config["DATASET_NAME"])
        plt.legend(loc="best")  # lower left upper left
        plt.grid(True)
        # plt.yscale("log")
        plt.tight_layout()
        plt.savefig(self.logdir + "/" + add_prefix(self.plot_type) + ".pdf", format="pdf", dpi=300, bbox_inches="tight")
        plt.show()
        plt.figure()