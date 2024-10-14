from typing import List, Tuple, Dict, Union, Any
from finol.evaluation_layer.economic_distiller import EconomicDistiller
from finol.evaluation_layer.metric_calculator import MetricCalculator
from finol.evaluation_layer.benchmark_loader import BenchmarkLoader
from finol.evaluation_layer.result_visualizer import ResultVisualizer
from finol.utils import load_config, send_message_dingding


class ModelEvaluator:
    """
    Class to evaluate the performance of a trained model using various metrics.
    This class facilitates the assessment of model effectiveness by comparing it against benchmarks,
    calculating performance metrics, and visualizing the results.

    :param load_dataset_output: Dictionary containing output from function :func:`~finol.data_layer.DatasetLoader.load_dataset`.
    :param train_model_output: Dictionary containing output from function :func:`~finol.optimization_layer.ModelTrainer.train_model`.
    """
    def __init__(self, load_dataset_output: Dict, train_model_output: Dict) -> None:
        self.config = load_config()
        self.load_dataset_output = load_dataset_output
        self.train_model_output = train_model_output

    def evaluate_model(self) -> Dict:
        """
        Evaluate the model based on the loaded dataset and trained model.

        :return: Dictionary containing the evaluation output of the model.
        """
        # Step 0: Distill the model if needed
        economic_distillation_output = None
        if self.config["INTERPRETABLE_ANALYSIS_CONFIG"]["INCLUDE_INTERPRETABILITY_ANALYSIS"] or self.config["INTERPRETABLE_ANALYSIS_CONFIG"]["INCLUDE_ECONOMIC_DISTILLATION"]:
            economic_distillation_output = EconomicDistiller(self.load_dataset_output, self.train_model_output).economic_distillation()
            print("economic_distillation_output")
            print(economic_distillation_output)

        # Step 1: Calculate the results of the model
        calculate_metric_output = MetricCalculator(self.load_dataset_output, self.train_model_output).calculate_metric()
        print("calculate_metric_output")
        print(calculate_metric_output)

        # Step 2: Calculate the results of the model
        load_benchmark_output = BenchmarkLoader(calculate_metric_output, economic_distillation_output).load_benchmark()

        # Step 3: Visualize the results
        visualize_result_output = ResultVisualizer(load_benchmark_output).visualize_result()

        # send_message_dingding(load_benchmark_output)
        evaluate_model_output = load_benchmark_output
        return evaluate_model_output

