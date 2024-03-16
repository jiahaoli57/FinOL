import time

import torch
import numpy as np
import matplotlib.pyplot as plt
from finol.evaluation_layer.metric_caculator import *
from finol.evaluation_layer.benchmark_loader import *
from finol.config import *


def evaluate_model(load_dataset_output, train_model_output):
    model = train_model_output["best_model"]
    logdir = train_model_output["logdir"]
    if model == None:
        model = torch.load(logdir + '/best_model_'+DATASET_NAME+'.pt')

    model.eval()

    # Training, validation and test sets
    # train_loader = load_dataset_output["train_loader"]
    # val_loader = load_dataset_output["val_loader"]
    test_loader = load_dataset_output["test_loader"]
    # NUM_TRAIN_PERIODS = load_dataset_output["NUM_TRAIN_PERIODS"]
    # NUM_VAL_PERIODS = load_dataset_output["NUM_VAL_PERIODS"]
    # NUM_TEST_PERIODS = load_dataset_output["NUM_TEST_PERIODS"]
    # NUM_ASSETS = load_dataset_output["NUM_ASSETS"]

    caculate_metric_output = caculate_metric(model, test_loader)
    print(caculate_metric_output)
    load_benchmark_output = load_benchmark(caculate_metric_output)
    evaluate_model_output = load_benchmark_output

    return evaluate_model_output

if __name__ == '__main__':
    from finol.data_layer.data_loader import *
    from finol.optimization_layer.model_trainer import *
    load_benchmark_output = load_benchmark()
    load_dataset_output = load_dataset()
    # train_model_output = train_model(load_dataset_output)
    train_model_output = {
        "last_model": None,
        "best_model": None,
        "logdir": ROOT_PATH + '/evaluation_layer/2024-03-13_17-47-22'
    }
    evaluate_model_output = evaluate_model(load_dataset_output, train_model_output)
