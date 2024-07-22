from finol.data_layer.DatasetLoader import DatasetLoader
from finol.optimization_layer.ModelTrainer import ModelTrainer
from finol.evaluation_layer.ModelEvaluator import ModelEvaluator


load_dataset_output = DatasetLoader().load_dataset()
train_model_output = ModelTrainer(load_dataset_output).train_model()
evaluate_model_output = ModelEvaluator(load_dataset_output, train_model_output).evaluate_model()