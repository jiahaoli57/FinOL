from finol.data_layer.dataset_loader import DatasetLoader
from finol.optimization_layer.model_trainer import ModelTrainer
from finol.evaluation_layer.model_evaluator import ModelEvaluator

load_dataset_output = DatasetLoader().load_dataset()
train_model_output = ModelTrainer(load_dataset_output).train_model()
evaluate_model_output = ModelEvaluator(load_dataset_output, train_model_output).evaluate_model()