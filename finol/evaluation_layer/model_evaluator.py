from finol.evaluation_layer.economic_distiller import EconomicDistiller
from finol.evaluation_layer.metric_caculator import MetricCaculator
from finol.evaluation_layer.benchmark_loader import BenchmarkLoader
from finol.evaluation_layer.result_visualizer import ResultVisualizer
from finol.utils import load_config, send_message_dingding


class ModelEvaluator:
    def __init__(self, load_dataset_output, train_model_output):
        self.config = load_config()
        self.load_dataset_output = load_dataset_output
        self.train_model_output = train_model_output

    def evaluate_model(self):
        # Step 0: Distill the model if needed
        economic_distiller_caculate_metric_output = None
        if self.config["INTERPRETABLE_ANALYSIS_CONFIG"]["INCLUDE_INTERPRETABILITY_ANALYSIS"] or self.config["INTERPRETABLE_ANALYSIS_CONFIG"]["INCLUDE_ECONOMIC_DISTILLATION"]:
            economic_distiller_caculate_metric_output = EconomicDistiller(self.load_dataset_output, self.train_model_output).economic_distillation()

        # Step 1: Calculate the results of the model
        caculate_metric_output = MetricCaculator(self.load_dataset_output, self.train_model_output).caculate_metric()

        # Step 2: Calculate the results of the model
        load_benchmark_output = BenchmarkLoader(caculate_metric_output, economic_distiller_caculate_metric_output).load_benchmark()

        # Step 3: Visualize the results
        ResultVisualizer(load_benchmark_output).visualize_result()

        send_message_dingding(load_benchmark_output)
        return load_benchmark_output


if __name__ == "__main__":
    from finol.data_layer import DatasetLoader
    from finol.utils import ROOT_PATH
    load_dataset_output = DatasetLoader().load_dataset()
    train_model_output = {
        # "logdir": ROOT_PATH + "/logdir/" + DATASET_NAME + "-" + MODEL_NAME
        "logdir": ROOT_PATH + "/logdir/2024-07-27_21-12-11"
    }
    evaluate_model_output = ModelEvaluator(load_dataset_output, train_model_output).evaluate_model()