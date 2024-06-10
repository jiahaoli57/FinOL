import time
import torch
import numpy as np
import pandas as pd
import torch.nn as nn

from captum.attr import Saliency
from finol.config import *
from finol.evaluation_layer.economic_distiller_metric_caculator import ed_caculate_metric
from finol.evaluation_layer.distiller_selector import select_distiller
from finol.utils import portfolio_selection, actual_portfolio_selection

import warnings
warnings.filterwarnings("ignore")


class ModifiedModel(nn.Module):
    def __init__(self, model):
        super(ModifiedModel, self).__init__()
        self.base_model = model

    def forward(self, x):
        x = self.base_model(x)
        x = actual_portfolio_selection(x)
        return x


def economic_distillation(load_dataset_output, train_model_output):
    logdir = train_model_output["logdir"]
    model = torch.load(logdir + '/best_model_' + DATASET_NAME + '.pt').to(DEVICE)
    model.eval()

    if INTERPRETABLE_ANALYSIS_CONFIG['Y_NAME'] == 'PORTFOLIOS':
        model = ModifiedModel(model).to(DEVICE)
        model.eval()

    test_loader = load_dataset_output['test_loader']
    NUM_TEST_PERIODS = load_dataset_output['NUM_TEST_PERIODS']
    NUM_ASSETS = load_dataset_output['NUM_ASSETS']
    NUM_FEATURES_ORIGINAL = load_dataset_output['NUM_FEATURES_ORIGINAL']
    DETAILED_NUM_FEATURES = load_dataset_output['DETAILED_NUM_FEATURES']
    WINDOW_SIZE = load_dataset_output['WINDOW_SIZE']
    OVERALL_FEATURE_LIST = load_dataset_output['OVERALL_FEATURE_LIST']
    DETAILED_FEATURE_LIST = load_dataset_output['DETAILED_FEATURE_LIST']

    coef_list = []
    intercept_list = []
    indices_list = []

    if INTERPRETABLE_ANALYSIS_CONFIG['INCLUDE_INTERPRETABILITY_ANALYSIS']:
        ig = Saliency(model)
        feature_att = torch.zeros(NUM_TEST_PERIODS, NUM_ASSETS, NUM_FEATURES_ORIGINAL)
        every_day_att = torch.zeros(NUM_TEST_PERIODS, NUM_FEATURES_ORIGINAL)

    for day, data in enumerate(test_loader):
        x_data, label = data

        if INTERPRETABLE_ANALYSIS_CONFIG['INCLUDE_INTERPRETABILITY_ANALYSIS']:
            # Calculate the feature attributions for each asset using saliency method
            for i in range(NUM_ASSETS):  # num_feats
                attributions = ig.attribute(x_data.float(), target=i, abs=False)
                attributions = attributions.view(1, NUM_ASSETS, WINDOW_SIZE, NUM_FEATURES_ORIGINAL)
                # attributions.shape = state.shape = torch.Size([1, NUM_ASSETS, WINDOW_SIZE, NUM_FEATURES_ORIGINAL])
                attributions = attributions[:, i, :, :]  # attributions_sum.shape: torch.Size([1, WINDOW_SIZE, NUM_FEATURES_ORIGINAL])
                attributions_mean = torch.mean(attributions, dim=(0, 1))  # [1, WINDOW_SIZE, NUM_FEATURES_ORIGINAL] -> [NUM_FEATURES_ORIGINAL]
                # print(attributions_sum.shape)
                feature_att[day, i, :] = attributions_mean
            every_day_att[day, :] = torch.mean(feature_att[day, :, :], dim=0)  # [NUM_ASSETS, NUM_FEATURES_ORIGINAL] -> [NUM_FEATURES_ORIGINAL]

        if INTERPRETABLE_ANALYSIS_CONFIG['INCLUDE_ECONOMIC_DISTILLATION']:
            final_scores = model(x_data.float())

            if INTERPRETABLE_ANALYSIS_CONFIG['PROP_DISTILLED_FEATURES'] != 1:
                _, indices = torch.topk(every_day_att[day, :], k=int(INTERPRETABLE_ANALYSIS_CONFIG['PROP_DISTILLED_FEATURES'] * NUM_FEATURES_ORIGINAL))  # 获取top k索引
            else:
                indices = torch.arange(0, NUM_FEATURES_ORIGINAL)

            flat_x_data = x_data.view(1, NUM_ASSETS, WINDOW_SIZE, NUM_FEATURES_ORIGINAL).squeeze(0)
            linear_x_data = torch.mean(flat_x_data[:, :, indices], dim=1)  # [1, NUM_ASSETS, WINDOW_SIZE, NUM_FEATURES_ORIGINAL] -> [NUM_ASSETS, NUM_FEATURES_ORIGINAL] -> np

            linear_x = linear_x_data.cpu().detach().numpy()
            linear_y = final_scores.squeeze(0).cpu().detach().numpy()

            linear_model = select_distiller()
            linear_model.fit(linear_x, linear_y)

            coef = np.zeros(NUM_FEATURES_ORIGINAL)  # Initializes a zero vector whose length is the number of features
            real_coef = linear_model.coef_  # Take the actual non-zero coefficient
            coef[indices] = real_coef  # Insert the actual nonzero coefficient into the corresponding position of the zero vector

            # Save to generate a linear portfolio when you iterate over the test set a second time
            coef_list.append(coef)
            intercept_list.append(linear_model.intercept_)
            indices_list.append(indices)

    if INTERPRETABLE_ANALYSIS_CONFIG['INCLUDE_INTERPRETABILITY_ANALYSIS']:
        # Mean the every_day_att across days
        every_day_att_mean = torch.mean(every_day_att, dim=0)
        print(every_day_att_mean)

        # Plot the feature attributions using matplotlib.pyplot
        fig = plt.figure(figsize=(9, 4))
        # Plot the average feature attribution for each feature across all time steps as a bar chart
        pre_num_features = 0
        for i, unit in enumerate(OVERALL_FEATURE_LIST):
            current_num_features = DETAILED_NUM_FEATURES[unit]
            mean_result = torch.mean(every_day_att_mean[pre_num_features: current_num_features+pre_num_features])
            plt.bar(unit, mean_result)
            pre_num_features = current_num_features

        plt.xlabel('Features')
        plt.ylabel('Importance of Features')
        plt.title(DATASET_NAME)
        plt.grid(True)
        plt.tight_layout()
        fig.autofmt_xdate()
        plt.savefig(logdir + '/' + MODEL_NAME + '_' + DATASET_NAME + '_INTERPRETABILITY_ANALYSIS.pdf',
                    format='pdf',
                    dpi=300,
                    bbox_inches='tight')
        plt.show()

    if INTERPRETABLE_ANALYSIS_CONFIG['INCLUDE_ECONOMIC_DISTILLATION']:
        portfolios = torch.zeros((NUM_TEST_PERIODS, NUM_ASSETS))
        labels = torch.zeros((NUM_TEST_PERIODS, NUM_ASSETS))
        start_time = time.time()
        for day, data in enumerate(test_loader):
            x_data, label = data

            coef = coef_list[day]
            intercept = intercept_list[day]
            indices = indices_list[day]

            flat_x_data = x_data.view(1, NUM_ASSETS, WINDOW_SIZE, NUM_FEATURES_ORIGINAL).squeeze(0)
            linear_x_data = torch.mean(flat_x_data, dim=1)  # [1, NUM_ASSETS, WINDOW_SIZE, NUM_FEATURES_ORIGINAL] -> [NUM_ASSETS, NUM_FEATURES_ORIGINAL]

            x_new = linear_x_data.cpu().detach().numpy()
            y_pred = np.dot(x_new, coef) + intercept

            # pred -> portfolio
            y_pred = torch.from_numpy(y_pred).to(DEVICE)
            # print(y_pred)
            if INTERPRETABLE_ANALYSIS_CONFIG["Y_NAME"] == "SCORES":
                portfolio = actual_portfolio_selection(y_pred.unsqueeze(0))
            elif INTERPRETABLE_ANALYSIS_CONFIG["Y_NAME"] == "PORTFOLIOS":
                positive_y_pred = torch.clamp(y_pred, min=1e-2)
                portfolio = positive_y_pred / torch.sum(positive_y_pred)

            labels[day, :] = label
            portfolios[day, :] = portfolio

        runtime = time.time() - start_time
        economic_distiller_caculate_metric_output = ed_caculate_metric(portfolios, labels, runtime)
        # print(economic_distiller_caculate_metric_output)

        # Convert coefficients to ndarray
        coef_array = np.array(coef_list)
        pre_num_features = 0
        mean_coef_array = np.zeros((NUM_TEST_PERIODS, len(OVERALL_FEATURE_LIST)))
        for i, unit in enumerate(OVERALL_FEATURE_LIST):
            current_num_features = DETAILED_NUM_FEATURES[unit]
            mean_result = np.mean(coef_array[:, pre_num_features: current_num_features + pre_num_features], axis=1)
            mean_coef_array[:, i] = mean_result
            pre_num_features = current_num_features

        # Calculate the statistical characteristics of time
        pd.set_option('display.max_column', None)
        print(pd.DataFrame(mean_coef_array).describe().round(4))

        # Draw a coefficient line chart
        plt.figure()
        for i in range(mean_coef_array.shape[1]):
            plt.plot(mean_coef_array[:, i], color='black', marker=MARKERS[i], markevery=MARKEVERY, alpha=ALPHA)
        plt.title(DATASET_NAME)
        if PLOT_CHINESE:
            plt.ylabel('特征重要程度')
            plt.xlabel('交易期')
        else:
            plt.ylabel('Coefficients')
            plt.xlabel('Trading Periods')
        plt.legend(OVERALL_FEATURE_LIST)
        plt.grid(True)
        plt.show()

        # Draw a coefficient heat map
        plt.figure(figsize=(9, 2.5))
        vmax = max(abs(np.max(mean_coef_array)), abs(np.min(mean_coef_array)))
        plt.imshow(mean_coef_array.T, aspect='auto', alpha=0.6, cmap='bwr', vmax=vmax, vmin=-vmax)
        plt.title(DATASET_NAME)
        if PLOT_CHINESE:
            plt.ylabel('特征')
            plt.xlabel('交易期')
        else:
            plt.ylabel('Feature')
            plt.xlabel('Trading Periods')

        OVERALL_FEATURE_LIST = []
        if PLOT_CHINESE:
            if FEATURE_ENGINEERING_CONFIG["INCLUDE_OHLCV_FEATURES"]:
                OVERALL_FEATURE_LIST.append('基础特征')
            if FEATURE_ENGINEERING_CONFIG["INCLUDE_OVERLAP_FEATURES"]:
                OVERALL_FEATURE_LIST.append('重叠特征')
            if FEATURE_ENGINEERING_CONFIG["INCLUDE_MOMENTUM_FEATURES"]:
                OVERALL_FEATURE_LIST.append('动量特征')
            if FEATURE_ENGINEERING_CONFIG["INCLUDE_VOLUME_FEATURES"]:
                OVERALL_FEATURE_LIST.append('量价特征')
            if FEATURE_ENGINEERING_CONFIG["INCLUDE_CYCLE_FEATURES"]:
                OVERALL_FEATURE_LIST.append('周期特征')
            if FEATURE_ENGINEERING_CONFIG["INCLUDE_PRICE_FEATURES"]:
                OVERALL_FEATURE_LIST.append('价格特征')
            if FEATURE_ENGINEERING_CONFIG["INCLUDE_VOLATILITY_FEATURES"]:
                OVERALL_FEATURE_LIST.append('波动特征')
            if FEATURE_ENGINEERING_CONFIG["INCLUDE_PATTERN_FEATURES"]:
                OVERALL_FEATURE_LIST.append('模式特征')
        else:
            if FEATURE_ENGINEERING_CONFIG["INCLUDE_OHLCV_FEATURES"]:
                OVERALL_FEATURE_LIST.append('OHLCV Features')
            if FEATURE_ENGINEERING_CONFIG["INCLUDE_OVERLAP_FEATURES"]:
                OVERALL_FEATURE_LIST.append('Overlap Features')
            if FEATURE_ENGINEERING_CONFIG["INCLUDE_MOMENTUM_FEATURES"]:
                OVERALL_FEATURE_LIST.append('Momentum Features')
            if FEATURE_ENGINEERING_CONFIG["INCLUDE_VOLUME_FEATURES"]:
                OVERALL_FEATURE_LIST.append('Volume Features')
            if FEATURE_ENGINEERING_CONFIG["INCLUDE_CYCLE_FEATURES"]:
                OVERALL_FEATURE_LIST.append('Cycle Features')
            if FEATURE_ENGINEERING_CONFIG["INCLUDE_PRICE_FEATURES"]:
                OVERALL_FEATURE_LIST.append('Price Features')
            if FEATURE_ENGINEERING_CONFIG["INCLUDE_VOLATILITY_FEATURES"]:
                OVERALL_FEATURE_LIST.append('Volatility Features')
            if FEATURE_ENGINEERING_CONFIG["INCLUDE_PATTERN_FEATURES"]:
                OVERALL_FEATURE_LIST.append('Pattern Features')

        plt.yticks(range(len(OVERALL_FEATURE_LIST)), OVERALL_FEATURE_LIST)
        plt.colorbar()
        plt.savefig(logdir + '/' + MODEL_NAME + '_' + DATASET_NAME + '_HEATMAP.pdf',
                    format='pdf',
                    dpi=300,
                    bbox_inches='tight')
        plt.show()
        return economic_distiller_caculate_metric_output
    return None