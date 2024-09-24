import matplotlib.pyplot as plt

from rich import print
from finol.evaluation_layer.benchmark_loader import BenchmarkLoader
from finol.evaluation_layer.distiller_selector import DistillerSelector
from finol.evaluation_layer.economic_distiller import EconomicDistiller
from finol.evaluation_layer.metric_calculator import MetricCalculator
from finol.evaluation_layer.model_evaluator import ModelEvaluator
from finol.evaluation_layer.result_visualizer import ResultVisualizer
from finol.utils import load_config

config = load_config()
if config["PLOT_LANGUAGE"].startswith('zh'):
    plt.rcParams["font.family"] = "Microsoft YaHei"

__all__ = [
    "BenchmarkLoader",
    "DistillerSelector",
    "EconomicDistiller",
    "MetricCalculator",
    "ModelEvaluator",
    "ResultVisualizer",
]
