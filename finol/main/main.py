import time

from finol.data_layer.DatasetLoader import DatasetLoader
from finol.optimization_layer.ModelTrainer import ModelTrainer
from finol.evaluation_layer.ModelEvaluator import ModelEvaluator
import json



# for i in range(1):
#     print('-'*60)
#     # 打开 JSON 文件并读取为 Python 字典
#     with open("../config.json", "r") as f:
#         config = json.load(f)
#
#     # 修改参数字典中的某个键的值
#     MANUAL_SEED = i
#     config["MANUAL_SEED"] = MANUAL_SEED
#
#     # 将修改后的字典写回 JSON 文件
#     with open("../config.json", "w") as f:
#         json.dump(config, f, indent=4)

load_dataset_output = DatasetLoader().load_dataset()
train_model_output = ModelTrainer(load_dataset_output).train_model()
evaluate_model_output = ModelEvaluator(load_dataset_output, train_model_output).evaluate_model()