import torch

from captum.attr import Saliency

from finol.config import *
from finol import *


def economic_distillation(load_dataset_output, train_model_output):
    logdir = train_model_output["logdir"]
    model = torch.load(logdir + '/best_model_' + DATASET_NAME + '.pt').to(DEVICE)
    model.eval()

    test_loader = load_dataset_output['test_loader']
    NUM_TEST_PERIODS = load_dataset_output['NUM_TEST_PERIODS']
    NUM_ASSETS = load_dataset_output['NUM_ASSETS']
    NUM_FEATURES_ORIGINAL = load_dataset_output['NUM_FEATURES_ORIGINAL']
    DETAILED_NUM_FEATURES = load_dataset_output['DETAILED_NUM_FEATURES']
    WINDOW_SIZE = load_dataset_output['WINDOW_SIZE']
    FEATURE_LIST = load_dataset_output['FEATURE_LIST']

    ig = Saliency(model)

    feature_att = torch.zeros(NUM_TEST_PERIODS, NUM_ASSETS, NUM_FEATURES_ORIGINAL)
    every_day_att = torch.zeros(NUM_TEST_PERIODS, NUM_FEATURES_ORIGINAL)

    for day, data in enumerate(test_loader):
        x_data, label = data

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

    # Mean the every_day_att across days
    every_day_att_mean = torch.mean(every_day_att, dim=0)
    print(every_day_att_mean)

    # Plot the feature attributions using matplotlib.pyplot
    fig = plt.figure(figsize=(9, 4))
    # 40
    # Plot the average feature attribution for each feature across all time steps as a bar chart
    pre_num_features = 0
    for i, unit in enumerate(FEATURE_LIST):
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
    plt.savefig(logdir + '/' + DATASET_NAME + '_Interpretability_Analysis.pdf',
                format='pdf',
                dpi=300,
                bbox_inches='tight')
    plt.show()