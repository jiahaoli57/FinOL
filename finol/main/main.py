from finol.data_layer.data_loader import load_dataset
from finol.optimization_layer.model_trainer import train_model
from finol.evaluation_layer.model_evaluator import evaluate_model
from finol.config import *


seed = MANUAL_SEED
load_dataset_output = load_dataset()
train_model_output = train_model(load_dataset_output, seed)
evaluate_model_output = evaluate_model(load_dataset_output, train_model_output)