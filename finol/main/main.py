from finol.data_layer.data_loader import *
from finol.optimization_layer.model_trainer import *
from finol.evaluation_layer.model_evaluator import *


load_dataset_output = load_dataset()
train_model_output = train_model(load_dataset_output)
evaluate_model_output = evaluate_model(load_dataset_output, train_model_output)