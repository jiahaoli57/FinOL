import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from finol.utils import load_config, add_prefix


def radar_factory(num_vars, frame="circle"):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {"circle", "polygon"}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = "radar"
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location("N")

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == "circle":
                return Circle((0.5, 0.5), 0.5)
            elif frame == "polygon":
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == "circle":
                return super()._gen_axes_spines()
            elif frame == "polygon":
                # spine_type must be "left"/"right"/"top"/"bottom"/"circle".
                spine = Spine(axes=self,
                              spine_type="circle",
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {"polar": spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


class ResultVisualizer:
    """
    Class to visualize the results of proposed data-driven OLPS method.

    :param: load_benchmark_output: Dictionary containing output from function :func:`~finol.evaluation_layer.BenchmarkLoader.load_benchmark`.
    """
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
        for plot_type in ["DCW", "DMDD", "TCW", "RADAR"]:
            self.plot_type = plot_type
            if plot_type == "DCW":
                self.plot_daily_cumulative_wealth()
            elif plot_type == "DMDD":
                self.plot_daily_maximum_drawdown()
            elif plot_type == "TCW":
                self.plot_transaction_costs_adjusted_wealth()
            elif plot_type == "RADAR":
                self.plot_radar_chart()

    def plot_daily_cumulative_wealth(self) -> None:
        """
        Visualize the daily cumulative wealth comparison between the model and the top baselines.

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

    def plot_daily_drawdown(self) -> None:
        """
        Visualize the daily drawdown comparison between the model and the top baselines.

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

    def plot_daily_maximum_drawdown(self) -> None:
        """
        Visualize the daily maximum drawdown comparison between the model and the top baselines.

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

    def plot_transaction_costs_adjusted_wealth(self) -> None:
        """
        Visualize the transaction costs adjusted wealth comparison between the model and the top baselines.

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

    def plot_radar_chart(self) -> None:
        """
        Visualize the radar chart comparison between the model and the top baselines.

        :return: None
        """
        final_profit_result = self.load_benchmark_output["final_profit_result"][["Metric"] + self.top_5_baselines].set_index("Metric")
        final_risk_result = self.load_benchmark_output["final_risk_result"][["Metric"] + self.top_5_baselines].set_index("Metric")

        df = pd.concat([final_profit_result, final_risk_result], ignore_index=False)
        num_columns = len(df.columns)

        data = []
        for strategy in self.top_5_baselines:
            _ = [df[strategy]["CW"], -df[strategy]["MDD"], -df[strategy]["VR"], df[strategy]["SR"], df[strategy]["APY"]]
            data += [_]

        # scale the data to [0, 1]
        data_array = np.array(data)
        min_val = np.min(data_array, axis=0)
        max_val = np.max(data_array, axis=0)
        scaled_data = (data_array - min_val) / (max_val - min_val)

        theta = radar_factory(num_vars=5, frame="polygon")
        labels = ["CW", "- MDD", "- VR", "SR", "APY"]

        fig, ax = plt.subplots(subplot_kw=dict(projection="radar"))
        colors = ["black"] * (num_columns - 1) + ["red"] * 1

        # plot the four cases from the example data on separate axes
        ax.set_rgrids([0.2, 0.4, 0.6, 0.8])

        for d, color, marker in zip(scaled_data, colors, self.config["MARKERS"]):
            ax.plot(theta, d, color=color, alpha=0.5, marker=marker)
            ax.fill(theta, d, facecolor=color, alpha=0.15, label="_nolegend_")
        ax.set_varlabels(labels)

        # add legend relative to top-left plot
        plt.title(self.config["DATASET_NAME"])
        ax.legend(df.columns, loc=(0.7, 0.75))
        plt.tight_layout()
        plt.savefig(self.logdir + "/" + add_prefix(self.plot_type) + ".pdf", format="pdf", dpi=300, bbox_inches="tight")
        plt.show()
