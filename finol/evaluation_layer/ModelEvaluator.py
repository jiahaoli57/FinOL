from finol.evaluation_layer.EconomicDistiller import EconomicDistiller
from finol.evaluation_layer.MetricCaculator import MetricCaculator
from finol.evaluation_layer.BenchmarkLoader import BenchmarkLoader
from finol.utils import ROOT_PATH, load_config, send_message_dingding


class ModelEvaluator:
    def __init__(self, load_dataset_output, train_model_output):
        self.config = load_config()
        self.load_dataset_output = load_dataset_output
        self.train_model_output = train_model_output

    def evaluate_model(self):
        economic_distiller_caculate_metric_output = None
        if self.config["INTERPRETABLE_ANALYSIS_CONFIG"]["INCLUDE_INTERPRETABILITY_ANALYSIS"] or self.config["INTERPRETABLE_ANALYSIS_CONFIG"]["INCLUDE_ECONOMIC_DISTILLATION"]:
            economic_distiller_caculate_metric_output = EconomicDistiller(self.load_dataset_output, self.train_model_output).economic_distillation()
        caculate_metric_output = MetricCaculator(self.load_dataset_output, self.train_model_output).caculate_metric()
        load_benchmark_output = BenchmarkLoader(caculate_metric_output, economic_distiller_caculate_metric_output).load_benchmark()

        send_message_dingding(load_benchmark_output)
        return load_benchmark_output


if __name__ == "__main__":
    from finol.data_layer.DatasetLoader import DatasetLoader
    load_dataset_output = DatasetLoader().load_dataset()
    train_model_output = {
        # "logdir": ROOT_PATH[:-5] + "/logdir/" + DATASET_NAME + "-" + MODEL_NAME
        "logdir": ROOT_PATH[:-5] + "/logdir/2024-07-16_18-55-57"
    }
    evaluate_model_output = ModelEvaluator(load_dataset_output, train_model_output).evaluate_model()