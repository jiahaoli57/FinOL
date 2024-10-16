import time
import torch
import warnings
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt

from captum.attr import Saliency
from typing import List, Tuple, Dict, Union
from finol.evaluation_layer.metric_calculator import MetricCalculator
from finol.evaluation_layer.distiller_selector import DistillerSelector
from finol.utils import load_config, portfolio_selection, actual_portfolio_selection, add_prefix

# plt.style.use("seaborn-paper")
# plt.rcParams["font.family"] = "Microsoft YaHei"
warnings.filterwarnings("ignore")
warnings.simplefilter(action="ignore", category=FutureWarning)


class ModifiedModel(nn.Module):
    def __init__(self, model):
        super(ModifiedModel, self).__init__()
        self.base_model = model

    def forward(self, x):
        x = self.base_model(x)
        x = actual_portfolio_selection(x)
        return x


class EconomicDistiller:
    """
    Class to implement economic distillation methods.

    :param load_dataset_output: Dictionary containing output from function :func:`~finol.data_layer.DatasetLoader.load_dataset`.
    :param train_model_output: Dictionary containing output from function :func:`~finol.optimization_layer.ModelTrainer.train_model`.
    """
    def __init__(self, load_dataset_output: Dict, train_model_output: Dict) -> None:
        self.config = load_config()
        self.load_dataset_output = load_dataset_output
        self.train_model_output = train_model_output
        self.logdir = train_model_output["logdir"]

        self.test_loader = load_dataset_output["test_loader"]
        self.num_test_periods = load_dataset_output["num_test_periods"]
        self.num_assets = load_dataset_output["num_assets"]
        self.num_features_original = load_dataset_output["num_features_original"]
        self.detailed_num_features = load_dataset_output["detailed_num_features"]
        self.window_size = load_dataset_output["window_size"]
        self.overall_feature_list = load_dataset_output["overall_feature_list"]
        self.detailed_feature_list = load_dataset_output["detailed_feature_list"]

    def economic_distillation(self) -> Union[Dict, None]:
        """
        Distill the trained model to a simple form using economic distillation techniques,
        and calculate various evaluation metrics based on the loaded dataset and
        the resulting economic distillation model.

        :return: None or dictionary containing calculated evaluation metrics for the economic distillation model.
        """
        model = torch.load(self.logdir + "/" + add_prefix("best_model.pt")).to(self.config["DEVICE"])
        model.eval()

        if self.config["INTERPRETABLE_ANALYSIS_CONFIG"]["Y_NAME"] == "Portfolios":
            model = ModifiedModel(model).to(self.config["DEVICE"])
            model.eval()

        coef_list = []
        intercept_list = []
        indices_list = []

        if self.config["INTERPRETABLE_ANALYSIS_CONFIG"]["INCLUDE_INTERPRETABILITY_ANALYSIS"]:
            ig = Saliency(model)
            feature_att = torch.zeros(self.num_test_periods, self.num_assets, self.num_features_original)
            every_day_att = torch.zeros(self.num_test_periods, self.num_features_original)

        for day, data in enumerate(self.test_loader):
            x_data, label = data

            if self.config["INTERPRETABLE_ANALYSIS_CONFIG"]["INCLUDE_INTERPRETABILITY_ANALYSIS"]:
                # Calculate the feature attributions for each asset using saliency method
                for i in range(self.num_assets):  # num_feats
                    attributions = ig.attribute(x_data.float(), target=i, abs=False)
                    attributions = attributions.view(1, self.num_assets, self.window_size, self.num_features_original)
                    # attributions.shape = state.shape = torch.Size([1, num_assets, window_size, num_features_original])
                    attributions = attributions[:, i, :, :]  # attributions_sum.shape: torch.Size([1, window_size, num_features_original])
                    attributions_mean = torch.mean(attributions, dim=(0, 1))  # [1, window_size, num_features_original] -> [num_features_original]
                    # print(attributions_sum.shape)
                    feature_att[day, i, :] = attributions_mean
                every_day_att[day, :] = torch.mean(feature_att[day, :, :], dim=0)  # [num_assets, num_features_original] -> [num_features_original]

            if self.config["INTERPRETABLE_ANALYSIS_CONFIG"]["INCLUDE_ECONOMIC_DISTILLATION"]:
                final_scores = model(x_data.float())

                if self.config["INTERPRETABLE_ANALYSIS_CONFIG"]["PROP_DISTILLED_FEATURES"] != 1:
                    _, indices = torch.topk(every_day_att[day, :], k=int(self.config["INTERPRETABLE_ANALYSIS_CONFIG"]["PROP_DISTILLED_FEATURES"] * self.num_features_original))  # 获取top k索引
                else:
                    indices = torch.arange(0, self.num_features_original)

                flat_x_data = x_data.view(1, self.num_assets, self.window_size, self.num_features_original).squeeze(0)
                linear_x_data = torch.mean(flat_x_data[:, :, indices], dim=1)  # [1, num_assets, window_size, num_features_original] -> [num_assets, num_features_original] -> np

                linear_x = linear_x_data.cpu().detach().numpy()
                linear_y = final_scores.squeeze(0).cpu().detach().numpy()

                linear_model = DistillerSelector().select_distiller()
                linear_model.fit(linear_x, linear_y)

                coef = np.zeros(self.num_features_original)  # Initializes a zero vector whose length is the number of features
                real_coef = linear_model.coef_  # Take the actual non-zero coefficient
                coef[indices] = real_coef  # Insert the actual nonzero coefficient into the corresponding position of the zero vector

                # Save to generate a linear portfolio when you iterate over the test set a second time
                coef_list.append(coef)
                intercept_list.append(linear_model.intercept_)
                indices_list.append(indices)

        if self.config["INTERPRETABLE_ANALYSIS_CONFIG"]["INCLUDE_INTERPRETABILITY_ANALYSIS"]:
            # Mean the every_day_att across days
            every_day_att_mean = torch.mean(every_day_att, dim=0)
            # print(every_day_att_mean)

            # Plot the feature attributions using matplotlib.pyplot
            fig = plt.figure(figsize=(9, 4))
            # Plot the average feature attribution for each feature across all time steps as a bar chart
            pre_num_features = 0
            for i, unit in enumerate(self.overall_feature_list):
                current_num_features = self.detailed_num_features[unit]
                mean_result = torch.mean(every_day_att_mean[pre_num_features: current_num_features+pre_num_features])
                plt.bar(unit, mean_result)
                print(f"unit, mean_result: {unit, mean_result}")
                pre_num_features = current_num_features

            plot_labels = {
                "en": ("Features", "Coefficients"),
                "zh_CN": ("特征", "特征重要程度"),
                "zh_TW": ("特徵", "特徵重要程度")
            }
            xlabel, ylabel = plot_labels[self.config["PLOT_LANGUAGE"]]
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(self.config["DATASET_NAME"])
            plt.grid(True)
            plt.tight_layout()
            fig.autofmt_xdate()
            plt.savefig(self.logdir + "/" + self.config["MODEL_NAME"] + "_" + self.config["DATASET_NAME"] + "_INTERPRETABILITY_ANALYSIS.pdf",
                        format="pdf",
                        dpi=300,
                        bbox_inches="tight")
            plt.show()

        if self.config["INTERPRETABLE_ANALYSIS_CONFIG"]["INCLUDE_ECONOMIC_DISTILLATION"]:
            portfolios = torch.zeros((self.num_test_periods, self.num_assets))
            labels = torch.zeros((self.num_test_periods, self.num_assets))
            start_time = time.time()
            for day, data in enumerate(self.test_loader):
                x_data, label = data

                coef = coef_list[day]
                intercept = intercept_list[day]
                indices = indices_list[day]

                flat_x_data = x_data.view(1, self.num_assets, self.window_size, self.num_features_original).squeeze(0)
                linear_x_data = torch.mean(flat_x_data, dim=1)  # [1, num_assets, window_size, num_features_original] -> [num_assets, num_features_original]

                x_new = linear_x_data.cpu().detach().numpy()
                y_pred = np.dot(x_new, coef) + intercept

                # pred -> portfolio
                y_pred = torch.from_numpy(y_pred).to(self.config["DEVICE"])
                # print(y_pred)
                if self.config["INTERPRETABLE_ANALYSIS_CONFIG"]["Y_NAME"] == "Scores":
                    portfolio = actual_portfolio_selection(y_pred.unsqueeze(0))
                elif self.config["INTERPRETABLE_ANALYSIS_CONFIG"]["Y_NAME"] == "Portfolios":
                    positive_y_pred = torch.clamp(y_pred, min=1e-2)
                    portfolio = positive_y_pred / torch.sum(positive_y_pred)

                labels[day, :] = label
                portfolios[day, :] = portfolio

            runtime = time.time() - start_time
            economic_distiller_calculate_metric_output = MetricCalculator(mode="ed").calculate_metric(portfolios, labels, runtime)
            # print(economic_distiller_calculate_metric_output)

            # Convert coefficients to ndarray
            coef_array = np.array(coef_list)
            pre_num_features = 0
            mean_coef_array = np.zeros((self.num_test_periods, len(self.overall_feature_list)))
            for i, unit in enumerate(self.overall_feature_list):
                current_num_features = self.detailed_num_features[unit]
                mean_result = np.mean(coef_array[:, pre_num_features: current_num_features + pre_num_features], axis=1)
                mean_coef_array[:, i] = mean_result
                pre_num_features = current_num_features

            # Calculate the statistical characteristics of time
            pd.set_option("display.max_column", None)
            print(pd.DataFrame(mean_coef_array).describe().round(4))

            # Draw a coefficient line chart
            plt.figure()
            for i in range(mean_coef_array.shape[1]):
                plt.plot(mean_coef_array[:, i], color="black", marker=self.config["MARKERS"][i], markevery=self.config["MARKEVERY"][self.config["DATASET_NAME"]], alpha=0.5)
            plt.title(self.config["DATASET_NAME"])
            plot_labels = {
                "en": ("Trading Periods", "Coefficients"),
                "zh_CN": ("交易期", "特征重要程度"),
                "zh_TW": ("交易期", "特徵重要程度")
            }
            xlabel, ylabel = plot_labels[self.config["PLOT_LANGUAGE"]]
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.legend(self.overall_feature_list)
            plt.grid(True)
            plt.show()

            # Draw a coefficient heat map
            plt.figure(figsize=(9, 2.5))
            vmax = max(abs(np.max(mean_coef_array)), abs(np.min(mean_coef_array)))
            plt.imshow(mean_coef_array.T, aspect="auto", alpha=0.6, cmap="bwr", vmax=vmax, vmin=-vmax)
            plt.title(self.config["DATASET_NAME"])
            plot_labels = {
                "en": ("Trading Periods", "Feature"),
                "zh_CN": ("交易期", "特征"),
                "zh_TW": ("交易期", "特徵")
            }
            xlabel, ylabel = plot_labels[self.config["PLOT_LANGUAGE"]]
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)

            overall_feature_list = []
            if self.config["PLOT_CHINESE"]:
                if self.config["FEATURE_ENGINEERING_CONFIG"]["INCLUDE_OHLCV_FEATURES"]:
                    overall_feature_list.append("基础特征")
                if self.config["FEATURE_ENGINEERING_CONFIG"]["INCLUDE_OVERLAP_FEATURES"]:
                    overall_feature_list.append("重叠特征")
                if self.config["FEATURE_ENGINEERING_CONFIG"]["INCLUDE_MOMENTUM_FEATURES"]:
                    overall_feature_list.append("动量特征")
                if self.config["FEATURE_ENGINEERING_CONFIG"]["INCLUDE_VOLUME_FEATURES"]:
                    overall_feature_list.append("量价特征")
                if self.config["FEATURE_ENGINEERING_CONFIG"]["INCLUDE_CYCLE_FEATURES"]:
                    overall_feature_list.append("周期特征")
                if self.config["FEATURE_ENGINEERING_CONFIG"]["INCLUDE_PRICE_FEATURES"]:
                    overall_feature_list.append("价格特征")
                if self.config["FEATURE_ENGINEERING_CONFIG"]["INCLUDE_VOLATILITY_FEATURES"]:
                    overall_feature_list.append("波动特征")
                if self.config["FEATURE_ENGINEERING_CONFIG"]["INCLUDE_PATTERN_FEATURES"]:
                    overall_feature_list.append("模式特征")
            else:
                if self.config["FEATURE_ENGINEERING_CONFIG"]["INCLUDE_OHLCV_FEATURES"]:
                    overall_feature_list.append("OHLCV Features")
                if self.config["FEATURE_ENGINEERING_CONFIG"]["INCLUDE_OVERLAP_FEATURES"]:
                    overall_feature_list.append("Overlap Features")
                if self.config["FEATURE_ENGINEERING_CONFIG"]["INCLUDE_MOMENTUM_FEATURES"]:
                    overall_feature_list.append("Momentum Features")
                if self.config["FEATURE_ENGINEERING_CONFIG"]["INCLUDE_VOLUME_FEATURES"]:
                    overall_feature_list.append("Volume Features")
                if self.config["FEATURE_ENGINEERING_CONFIG"]["INCLUDE_CYCLE_FEATURES"]:
                    overall_feature_list.append("Cycle Features")
                if self.config["FEATURE_ENGINEERING_CONFIG"]["INCLUDE_PRICE_FEATURES"]:
                    overall_feature_list.append("Price Features")
                if self.config["FEATURE_ENGINEERING_CONFIG"]["INCLUDE_VOLATILITY_FEATURES"]:
                    overall_feature_list.append("Volatility Features")
                if self.config["FEATURE_ENGINEERING_CONFIG"]["INCLUDE_PATTERN_FEATURES"]:
                    overall_feature_list.append("Pattern Features")

            plt.yticks(range(len(overall_feature_list)), overall_feature_list)
            plt.colorbar()
            plt.savefig(self.logdir + "/" + self.config["MODEL_NAME"] + "_" + self.config["DATASET_NAME"] + "_HEATMAP.pdf",
                        format="pdf",
                        dpi=300,
                        bbox_inches="tight")
            plt.show()
            return economic_distiller_calculate_metric_output
        return None