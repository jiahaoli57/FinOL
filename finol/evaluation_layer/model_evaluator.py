from finol.evaluation_layer.economic_distiller import economic_distillation
from finol.evaluation_layer.metric_caculator import caculate_metric
from finol.evaluation_layer.benchmark_loader import load_benchmark
from finol.config import *
from finol.utils import send_message_dingding


def evaluate_model(load_dataset_output, train_model_output):
    economic_distiller_caculate_metric_output = None
    if INTERPRETABLE_ANALYSIS_CONFIG['INCLUDE_INTERPRETABILITY_ANALYSIS'] or INTERPRETABLE_ANALYSIS_CONFIG['INCLUDE_ECONOMIC_DISTILLATION']:
        economic_distiller_caculate_metric_output = economic_distillation(load_dataset_output, train_model_output)
    caculate_metric_output = caculate_metric(load_dataset_output, train_model_output)
    load_benchmark_output = load_benchmark(caculate_metric_output, economic_distiller_caculate_metric_output)

    send_message_dingding(load_benchmark_output)
    return load_benchmark_output


if __name__ == '__main__':
    from finol.data_layer.data_loader import load_dataset
    load_dataset_output = load_dataset()
    train_model_output = {
        # "logdir": ROOT_PATH[:-5] + '/logdir/' + DATASET_NAME + '-' + MODEL_NAME
        "logdir": ROOT_PATH[:-5] + '/logdir/Nasdaq-100-LSRE-CAAN'
    }
    evaluate_model_output = evaluate_model(load_dataset_output, train_model_output)