from finol.data_layer.data_loader import *
from finol.optimization_layer.model_trainer import *
from finol.evaluation_layer.economic_distiller import *
from finol.evaluation_layer.metric_caculator import *
from finol.evaluation_layer.benchmark_loader import *
from finol.config import *


def evaluate_model(load_dataset_output, train_model_output):
    if INTERPRETABLE_ANALYSIS_CONFIG['INCLUDE_INTERPRETABILITY_ANALYSIS']:
        economic_distillation(load_dataset_output, train_model_output)
    caculate_metric_output = caculate_metric(train_model_output, load_dataset_output)
    load_benchmark_output = load_benchmark(caculate_metric_output)

    return load_benchmark_output


if __name__ == '__main__':
    # load_benchmark_output = load_benchmark()
    load_dataset_output = load_dataset()
    train_model_output = {
        "logdir": ROOT_PATH[:-5] + '/logdir/2024-04-16_19-24-49_DJIA_WEEKLY'
    }
    evaluate_model_output = evaluate_model(load_dataset_output, train_model_output)
