import os
import json
import time

ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
PARENT_PATH = os.path.dirname(ROOT_PATH)

from rich import print

# with open(ROOT_PATH + "/config.json", "r") as f:
#     config = json.load(f)
#
# for key, value in config.items():
#     if not key.startswith('_'):  # 跳过注释键
#         globals()[key] = value

# print(f"GET_LATEST_FINOL: {globals()['GET_LATEST_FINOL']}")
# print(f"DATASET_NAME: {globals()['DATASET_NAME']}")

# GET_LATEST_FINOL = False
# TUTORIAL_MODE = True
# TUTORIAL_NAME = "TUTORIAL_4"
#
# ##########################################################################################
# ############################  Parameters related to data_layer ###########################
# ##########################################################################################
# LOAD_DATALOADER = True
# DATASET_NAME = "DJIA"  # Available options: NYSE(O), NYSE(N), DJIA, SP500, TSE, SSE, HSI, CMEG, CRYPTO, TUTORIAL, DJIA(N), Nasdaq-100
# DATASET_SPLIT_CONFIG = {
#     "NYSE(O)": {
#         "TRAIN_START_TIMESTAMP": "1962-07-03",
#         "TRAIN_END_TIMESTAMP": "1976-01-21",  # [3390 rows x 5 columns]
#         "VAL_START_TIMESTAMP": "1976-01-22",
#         "VAL_END_TIMESTAMP": "1980-07-11",  # [1130 rows x 5 columns]
#         "TEST_START_TIMESTAMP": "1980-07-14",
#         "TEST_END_TIMESTAMP": "1984-12-31"  # [1131 rows x 5 columns]
#     },
#     "NYSE(N)": {
#         "TRAIN_START_TIMESTAMP": "1985-01-02",
#         "TRAIN_END_TIMESTAMP": "2000-04-06",  # [3858 rows x 5 columns]
#         "VAL_START_TIMESTAMP": "2000-04-07",
#         "VAL_END_TIMESTAMP": "2005-05-20",  # [1286 rows x 5 columns]
#         "TEST_START_TIMESTAMP": "2005-05-23",
#         "TEST_END_TIMESTAMP": "2010-06-30"  # [1286 rows x 5 columns]
#     },
#     "DJIA": {
#         "TRAIN_START_TIMESTAMP": "2001-01-14",
#         "TRAIN_END_TIMESTAMP": "2002-04-01",  # [300 rows x 5 columns]
#         "VAL_START_TIMESTAMP": "2002-04-02",
#         "VAL_END_TIMESTAMP": "2002-08-21",  # [100 rows x 5 columns]
#         "TEST_START_TIMESTAMP": "2002-08-22",
#         "TEST_END_TIMESTAMP": "2003-01-14"  # [100 rows x 5 columns]
#     },
#     "SP500": {
#         "TRAIN_START_TIMESTAMP": "1998-01-02",
#         "TRAIN_END_TIMESTAMP": "2001-01-12",  # [756 rows x 5 columns]
#         "VAL_START_TIMESTAMP": "2001-01-16",
#         "VAL_END_TIMESTAMP": "2002-01-25",  # [256 rows x 5 columns]
#         "TEST_START_TIMESTAMP": "2002-01-28",
#         "TEST_END_TIMESTAMP": "2003-01-31"  # [256 rows x 5 columns]
#     },
#     "TSE": {
#         "TRAIN_START_TIMESTAMP": "1995-01-12",
#         "TRAIN_END_TIMESTAMP": "1997-05-28",  # [600 rows x 5 columns]
#         "VAL_START_TIMESTAMP": "1997-05-29",
#         "VAL_END_TIMESTAMP": "1998-03-13",  # [200 rows x 5 columns]
#         "TEST_START_TIMESTAMP": "1998-03-16",
#         "TEST_END_TIMESTAMP": "1998-12-31"  # [201 rows x 5 columns]
#     },
#     "SSE": {
#         "TRAIN_START_TIMESTAMP": "2010-07-05",
#         "TRAIN_END_TIMESTAMP": "2018-04-09",  # [406 rows x 5 columns]
#         "VAL_START_TIMESTAMP": "2018-04-16",
#         "VAL_END_TIMESTAMP": "2020-11-16",  # [136 rows x 5 columns]
#         "TEST_START_TIMESTAMP": "2020-11-23",
#         "TEST_END_TIMESTAMP": "2023-06-26"  # [136 rows x 5 columns]
#     },
#     "CRYPTO": {
#         "TRAIN_START_TIMESTAMP": "2017-11-09",
#         "TRAIN_END_TIMESTAMP": "2021-08-22",  # [1383 rows x 5 columns]
#         "VAL_START_TIMESTAMP": "2021-08-23",
#         "VAL_END_TIMESTAMP": "2022-11-26",  # [461 rows x 5 columns]
#         "TEST_START_TIMESTAMP": "2022-11-27",
#         "TEST_END_TIMESTAMP": "2024-03-01"  # [461 rows x 5 columns]
#     }
# }
# DATASET_SPLIT_CONFIG["HSI"] = DATASET_SPLIT_CONFIG["SSE"].copy()
# DATASET_SPLIT_CONFIG["CMEG"] = DATASET_SPLIT_CONFIG["SSE"].copy()
# DATASET_SPLIT_CONFIG["DJIA(N)"] = DATASET_SPLIT_CONFIG["SSE"].copy()
# DATASET_SPLIT_CONFIG["Nasdaq-100"] = DATASET_SPLIT_CONFIG["SSE"].copy()
#
# FEATURE_ENGINEERING_CONFIG = {
#     "INCLUDE_OHLCV_FEATURES": True,
#     "INCLUDE_OVERLAP_FEATURES": True,
#     "INCLUDE_MOMENTUM_FEATURES": True,
#     "INCLUDE_VOLUME_FEATURES": True,
#     "INCLUDE_CYCLE_FEATURES": True,
#     "INCLUDE_PRICE_FEATURES": True,
#     "INCLUDE_VOLATILITY_FEATURES": True,
#     "INCLUDE_PATTERN_FEATURES": True
# }
# DATA_AUGMENTATION_CONFIG = {
#     "WINDOW_DATA": {
#         "INCLUDE_WINDOW_DATA": True,
#         "WINDOW_SIZE": 50
#     }
# }
# SCALER = "WindowStandardScaler"  # None, StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler,
# # WindowStandardScaler, WindowMinMaxScaler, WindowMaxAbsScaler, WindowRobustScaler
# BATCH_SIZE = {
#     "NYSE(O)": 128,
#     "NYSE(N)": 128,
#     "DJIA": 64,
#     "SP500": 64,
#     "TSE": 64,
#     "SSE": 128,
#     "CRYPTO": 128,
# }
# BATCH_SIZE["HSI"] = BATCH_SIZE["SSE"]
# BATCH_SIZE["CMEG"] = BATCH_SIZE["SSE"]
# BATCH_SIZE["DJIA(N)"] = BATCH_SIZE["SSE"]
# BATCH_SIZE["Nasdaq-100"] = BATCH_SIZE["SSE"]
#
# DAYS = 52 if DATASET_NAME in ["SSE", "HSI", "CMEG", "DJIA(N)", 'Nasdaq-100'] else 252
#
# ##########################################################################################
# ########################### Parameters related to model_layer ############################
# ##########################################################################################
# MODEL_NAME = "LSRE-CAAN"  # AlphaPortfolio, CNN, DNN, LSRE-CAAN, LSTM, RNN, Transformer
# MODEL_CONFIG = {
#     "DNN": {
#         "NUM_LAYERS": 4,
#         "HIDDEN_SIZE": 32
#     },
#     "RNN": {
#         "NUM_LAYERS": 1,
#         "HIDDEN_SIZE": 32
#     },
#     "LSTM": {
#         "NUM_LAYERS": 4,
#         "HIDDEN_SIZE": 32
#     },
#     "CNN": {
#         "OUT_CHANNELS": 128,
#         "KERNEL_SIZE": 3,
#         "STRIDE": 1,
#         "HIDDEN_SIZE": 32,
#     },
#     "Transformer": {
#         "NUM_LAYERS": 1,
#         "NUM_HEADS": 1,
#         "HIDDEN_SIZE": 32,
#     },
#     "LSRE-CAAN": {
#         "NUM_LAYERS": 1,  # paper setting: 1
#         "NUM_LATENTS": 1,  # paper setting: 1
#         "LATENT_DIM": 32,  # paper setting: 32
#         "CROSS_HEADS": 1,  # paper setting: 1
#         "LATENT_HEADS": 1,  # paper setting: 1
#         "CROSS_DIM_HEAD": 64,  # paper setting: 64
#         "LATENT_DIM_HEAD": 32,  # paper setting: 32
#         "DROPOUT": 0.,  # paper setting: 0
#     },
#     "AlphaPortfolio": {
#         "DIM_EMBEDDING": 256,  # paper setting: 256
#         "DIM_FEEDFORWARD": 1024,  # paper setting: 1024
#         "NUM_HEADS": 4,  # paper setting: 4
#         "NUM_LAYERS": 1,  # paper setting: 1
#         "DROPOUT": 0.2,  # paper setting: 0.2
#     },
# }
# MODEL_CONFIG["LSRE-CAAN-d"] = MODEL_CONFIG["LSRE-CAAN"].copy()
# MODEL_CONFIG["LSRE-CAAN-dd"] = MODEL_CONFIG["LSRE-CAAN"].copy()
#
# MODEL_CONFIG_SPACE = {
#     'DNN': {
#         'NUM_LAYERS': {'type': 'int', 'range': (1, 12), 'step': 1},
#         'HIDDEN_SIZE': {'type': 'int', 'range': (32, 256), 'step': 32},
#         'DROPOUT': {'type': 'float', 'range': (0, 0.5), 'step': 0.05},
#     },
#     'Transformer': {
#         'NUM_LAYERS': {'type': 'int', 'range': (1, 12), 'step': 1},
#         'NUM_HEADS': {'type': 'int', 'range': (1, 12), 'step': 1},
#         'HIDDEN_SIZE': {'type': 'int', 'range': (32, 256), 'step': 32},
#         'DROPOUT': {'type': 'float', 'range': (0, 0.5), 'step': 0.05},
#     },
#     'LSRE-CAAN': {
#         'NUM_LAYERS': {'type': 'int', 'range': (1, 12), 'step': 1},
#         'NUM_LATENTS': {'type': 'int', 'range': (1, DATA_AUGMENTATION_CONFIG.get("WINDOW_DATA")["WINDOW_SIZE"]), 'step': 1},
#         'LATENT_DIM': {'type': 'int', 'range': (32, 256), 'step': 32},
#         'CROSS_HEADS': {'type': 'int', 'range': (1, 12), 'step': 1},
#         'LATENT_HEADS': {'type': 'int', 'range': (1, 12), 'step': 1},
#         'CROSS_DIM_HEAD': {'type': 'int', 'range': (32, 256), 'step': 32},
#         'LATENT_DIM_HEAD': {'type': 'int', 'range': (32, 256), 'step': 32},
#         'DROPOUT': {'type': 'float', 'range': (0, 0.5), 'step': 0.05},
#     },
#     "AlphaPortfolio": {
#         'DIM_EMBEDDING': {'type': 'int', 'range': (32, 256), 'step': 32},
#         'DIM_FEEDFORWARD': {'type': 'int', 'range': (32, 256), 'step': 32},
#         'NUM_HEADS': {'type': 'int', 'range': (1, 12), 'step': 1},
#         'NUM_LAYERS': {'type': 'int', 'range': (1, 12), 'step': 1},
#         'DROPOUT': {'type': 'float', 'range': (0, 0.5), 'step': 0.05},
#     },
# }
# MODEL_CONFIG_SPACE["RNN"] = MODEL_CONFIG_SPACE["DNN"].copy()
# MODEL_CONFIG_SPACE["LSTM"] = MODEL_CONFIG_SPACE["DNN"].copy()
# MODEL_CONFIG_SPACE["LSRE-CAAN-d"] = MODEL_CONFIG_SPACE["LSRE-CAAN"].copy()
# MODEL_CONFIG_SPACE["LSRE-CAAN-dd"] = MODEL_CONFIG_SPACE["LSRE-CAAN"].copy()
#
# DROPOUT = 0.1
# PROP_WINNERS = 0.5
#
# ##########################################################################################
# ######################## Parameters related to optimization_layer ########################
# ##########################################################################################
# TUNE_PARAMETERS = True
# MANUAL_SEED = 0  # 3442
# OPTIMIZER_NAME = "Lamb"
# LEARNING_RATE = 5e-4
# CRITERION_NAME = "LOG_WEALTH_L2_DIVERSIFICATION"  # LOG_WEALTH, LOG_WEALTH_L2_DIVERSIFICATION, LOG_WEALTH_L2_CONCENTRATION
# LAMBDA_L2 = 5e-4
# DEVICE = "cuda"
# NUM_EPOCHES = 1000
# SAVE_EVERY = 1
# PLOT_DYNAMIC_LOSS = True
#
#
# ##########################################################################################
# ######################### Parameters related to evaluation_layer #########################
# ##########################################################################################
# PLOT_CHINESE = False
# INTERPRETABLE_ANALYSIS_CONFIG = {
#     "INCLUDE_INTERPRETABILITY_ANALYSIS": False,
#     "INCLUDE_ECONOMIC_DISTILLATION": False,
#     "PROP_DISTILLED_FEATURES": 0.7,
#     "Y_NAME": "PORTFOLIOS",  # SCORES, PORTFOLIOS
#     "DISTILLER_NAME": "Lasso"
# }
# MARKERS = ['o', '^', '<', '>', 's', 'p', 'h', 'H', 'D', 'd']
# MARKEVERY = {
#     'NYSE(O)': 50,
#     'NYSE(N)': 55,
#     'DJIA': 3,
#     'SP500': 7,
#     'TSE': 6,
#     'SSE': 4,
#     'CRYPTO': 13,
#     }
# MARKEVERY["HSI"] = MARKEVERY["SSE"]
# MARKEVERY["CMEG"] = MARKEVERY["SSE"]
# MARKEVERY["DJIA(N)"] = MARKEVERY["SSE"]
# MARKEVERY["Nasdaq-100"] = MARKEVERY["SSE"]
# MARKEVERY = MARKEVERY.get(DATASET_NAME, 10)
#
# ALPHA = 0.5
# METRIC_CONFIG = {
#     "PRACTICAL_METRICS": {
#         "TRANSACTIOS_COSTS_RATE": 0.5/100,
#         "TRANSACTIOS_COSTS_RATE_INTERVAL": 1/1000,
#     }
# }
# BENCHMARK_BASELINE = ["Market", "Best", "UCRP", "BCRP"]
# FOLLOW_THE_WINNER = ["UP", "EG", "SCRP", "PPT", "SSPO"]
# FOLLOW_THE_LOSER = ["ANTI1", "ANTI2", "PAMR", "CWMR-Var", "CWMR-Stdev", "OLMAR-S", "OLMAR-E", "RMR", "RPRT"]
# PATTERN_MATCHING = ["AICTR", "KTPT"]
# META_LEARNING = ["SP", "ONS", "GRW", "WAAS", "CW-OGD"]
#
# # PLOT_ALL_1 = BENCHMARK_BASELINE + [MODEL_NAME]
# # PLOT_ALL_2 = FOLLOW_THE_WINNER + [MODEL_NAME]
# # PLOT_ALL_3 = FOLLOW_THE_LOSER + [MODEL_NAME]
# # PLOT_ALL_4 = PATTERN_MATCHING + [MODEL_NAME]
# # PLOT_ALL_5 = META_LEARNING + [MODEL_NAME]
#
# PLOT_ALL_1 = ["Market", "SCRP", "OLMAR-E", "RMR", "CW-OGD"] + [MODEL_NAME]
# # PLOT_ALL_2 = ["Market", "UP", "EG", "PAMR", "OLMAR-E", "RPRT", "KTPT", "CW-OGD"] + [MODEL_NAME]
# # PLOT_ALL_3 = ["Market", "UP", "EG", "PAMR", "OLMAR-E", "RPRT", "KTPT", "CW-OGD"] + [MODEL_NAME]
# # PLOT_ALL_4 = ["Market", "UP", "EG", "PAMR", "OLMAR-E", "RPRT", "KTPT", "CW-OGD"] + [MODEL_NAME]
# # PLOT_ALL_5 = ["Market", "UP", "EG", "PAMR", "OLMAR-E", "RPRT", "KTPT", "CW-OGD"] + [MODEL_NAME]
#
# if INTERPRETABLE_ANALYSIS_CONFIG['INCLUDE_ECONOMIC_DISTILLATION']:
#     PLOT_ALL_1 = PLOT_ALL_1 + [MODEL_NAME + ' (ED)']
#     # PLOT_ALL_2 = PLOT_ALL_2 + [MODEL_NAME + ' (ED)']
#     # PLOT_ALL_3 = PLOT_ALL_3 + [MODEL_NAME + ' (ED)']
#     # PLOT_ALL_4 = PLOT_ALL_4 + [MODEL_NAME + ' (ED)']
#     # PLOT_ALL_5 = PLOT_ALL_5 + [MODEL_NAME + ' (ED)']
#
# ################################################################################################################
# ############################################# FOR TUTORIAL ONLY ################################################
# ################################################################################################################
# if TUTORIAL_MODE:
#     LOAD_DATALOADER = True
#     DATASET_NAME = "DJIA"
#     FEATURE_ENGINEERING_CONFIG = {
#         "INCLUDE_OHLCV_FEATURES": True,
#         "INCLUDE_OVERLAP_FEATURES": True,
#         "INCLUDE_MOMENTUM_FEATURES": True,
#         "INCLUDE_VOLUME_FEATURES": True,
#         "INCLUDE_CYCLE_FEATURES": True,
#         "INCLUDE_PRICE_FEATURES": True,
#         "INCLUDE_VOLATILITY_FEATURES": True,
#         "INCLUDE_PATTERN_FEATURES": True
#     }
#     DATA_AUGMENTATION_CONFIG = {
#         "WINDOW_DATA": {
#             "INCLUDE_WINDOW_DATA": True,
#             "WINDOW_SIZE": 10
#         }
#     }
#     SCALER = "WindowStandardScaler"  # None, StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler,
#     # WindowStandardScaler, WindowMinMaxScaler, WindowMaxAbsScaler, WindowRobustScaler
#     BATCH_SIZE = {
#         "NYSE(O)": 128,
#         "NYSE(N)": 128,
#         "DJIA": 32,
#         "SP500": 64,
#         "TSE": 64,
#         "SSE": 32,
#         "HSI": 32,
#         "CMEG": 32,
#         "CRYPTO": 128,
#     }
#
#     # Parameters related to model_layer
#     MODEL_NAME = 'LSRE-CAAN'
#     MODEL_CONFIG = {
#         'DNN': {
#             'NUM_LAYERS': 1,
#             "HIDDEN_SIZE": 64,
#             "DROPOUT":  0.0
#         },
#         "RNN": {
#             "NUM_LAYERS": 1,
#             "HIDDEN_SIZE": 32,
#             "DROPOUT": 0.25,
#         },
#         "LSTM": {
#             "NUM_LAYERS": 4,
#             "HIDDEN_SIZE": 32,
#             "DROPOUT": 0.25,
#         },
#         "CNN": {
#             "OUT_CHANNELS": 128,
#             "KERNEL_SIZE": 3,
#             "STRIDE": 1,
#             "HIDDEN_SIZE": 32,
#             "DROPOUT": 0.25,
#         },
#         "Transformer": {
#             "NUM_LAYERS": 1,
#             "NUM_HEADS": 1,
#             "HIDDEN_SIZE": 32,
#             "DROPOUT": 0.25,
#         },
#         "LSRE-CAAN": {
#             "NUM_LAYERS": 1,  # paper setting: 1
#             "NUM_LATENTS": 1,  # paper setting: 1
#             "LATENT_DIM": 32,  # paper setting: 32
#             "CROSS_HEADS": 1,  # paper setting: 1
#             "LATENT_HEADS": 1,  # paper setting: 1
#             "CROSS_DIM_HEAD": 64,  # paper setting: 64
#             "LATENT_DIM_HEAD": 32,  # paper setting: 32
#             "DROPOUT": 0.,  # paper setting: 0
#         },
#         "AlphaPortfolio": {
#             "DIM_EMBEDDING": 256,  # paper setting: 256
#             "DIM_FEEDFORWARD": 1024,  # paper setting: 1024
#             "NUM_HEADS": 4,  # paper setting: 4
#             "NUM_LAYERS": 1,  # paper setting: 1
#             "DROPOUT": 0.2,  # paper setting: 0.2
#         },
#     }
#     MODEL_CONFIG_SPACE = {
#         'DNN': {
#             'NUM_LAYERS': {'type': 'int', 'range': (1, 12), 'step': 1},
#             'HIDDEN_SIZE': {'type': 'int', 'range': (32, 256), 'step': 32},
#             'DROPOUT': {'type': 'float', 'range': (0, 0.5), 'step': 0.05},
#         },
#         'Transformer': {
#             'NUM_LAYERS': {'type': 'int', 'range': (1, 12), 'step': 1},
#             'NUM_HEADS': {'type': 'int', 'range': (1, 12), 'step': 1},
#             'HIDDEN_SIZE': {'type': 'int', 'range': (32, 256), 'step': 32},
#             'DROPOUT': {'type': 'float', 'range': (0, 0.5), 'step': 0.05},
#         },
#         'LSRE-CAAN': {
#             'NUM_LAYERS': {'type': 'int', 'range': (1, 12), 'step': 1},
#             'NUM_LATENTS': {'type': 'int', 'range': (1, DATA_AUGMENTATION_CONFIG.get("WINDOW_DATA")["WINDOW_SIZE"]), 'step': 1},
#             'LATENT_DIM': {'type': 'int', 'range': (32, 256), 'step': 32},
#             'CROSS_HEADS': {'type': 'int', 'range': (1, 12), 'step': 1},
#             'LATENT_HEADS': {'type': 'int', 'range': (1, 12), 'step': 1},
#             'CROSS_DIM_HEAD': {'type': 'int', 'range': (32, 256), 'step': 32},
#             'LATENT_DIM_HEAD': {'type': 'int', 'range': (32, 256), 'step': 32},
#             'DROPOUT': {'type': 'float', 'range': (0, 0.5), 'step': 0.05},
#         },
#         "AlphaPortfolio": {
#             'DIM_EMBEDDING': {'type': 'int', 'range': (32, 256), 'step': 32},
#             'DIM_FEEDFORWARD': {'type': 'int', 'range': (32, 256), 'step': 32},
#             'NUM_HEADS': {'type': 'int', 'range': (1, 12), 'step': 1},
#             'NUM_LAYERS': {'type': 'int', 'range': (1, 12), 'step': 1},
#             'DROPOUT': {'type': 'float', 'range': (0, 0.5), 'step': 0.05},
#         },
#     }
#     MODEL_CONFIG_SPACE["RNN"] = MODEL_CONFIG_SPACE["DNN"].copy()
#     MODEL_CONFIG_SPACE["LSTM"] = MODEL_CONFIG_SPACE["DNN"].copy()
#     MODEL_CONFIG_SPACE["LSRE-CAAN-d"] = MODEL_CONFIG_SPACE["LSRE-CAAN"].copy()
#     MODEL_CONFIG_SPACE["LSRE-CAAN-dd"] = MODEL_CONFIG_SPACE["LSRE-CAAN"].copy()
#     PROP_WINNERS = 1
#
#     # Parameters related to optimization_layer
#     TUNE_PARAMETERS = True
#     MANUAL_SEED = 0  # 3442
#     OPTIMIZER_NAME = "Adam"
#     LEARNING_RATE = 5e-4
#     CRITERION_NAME = "LOG_WEALTH"  # LOG_WEALTH, LOG_WEALTH_L2_DIVERSIFICATION, LOG_WEALTH_L2_CONCENTRATION
#     LAMBDA_L2 = 5e-4
#     DEVICE = "cuda"
#     NUM_EPOCHES = 100
#     SAVE_EVERY = 1
#     PLOT_DYNAMIC_LOSS = True
#
#     # Parameters related to evaluation_layer
#     METRIC_CONFIG = {
#         "INCLUDE_PROFIT_METRICS": True,
#         "INCLUDE_RISK_METRICS": True,
#         "PRACTICAL_METRICS": {
#             "INCLUDE_PRACTICAL_METRICS": True,
#             "TRANSACTIOS_COSTS_RATE": 1 / 100,
#             "TRANSACTIOS_COSTS_RATE_INTERVAL": 1 / 1000,
#         }
#     }
#
#     PLOT_ALL_1 = ["Market", "SCRP", "OLMAR-E", "RMR", "CW-OGD"] + [MODEL_NAME]
#     # PLOT_ALL_2 = ["Market", "UP", "EG", "PAMR", "OLMAR-E", "RPRT", "KTPT", "CW-OGD"] + [MODEL_NAME]
#     # PLOT_ALL_3 = ["Market", "UP", "EG", "PAMR", "OLMAR-E", "RPRT", "KTPT", "CW-OGD"] + [MODEL_NAME]
#     # PLOT_ALL_4 = ["Market", "UP", "EG", "PAMR", "OLMAR-E", "RPRT", "KTPT", "CW-OGD"] + [MODEL_NAME]
#     # PLOT_ALL_5 = ["Market", "UP", "EG", "PAMR", "OLMAR-E", "RPRT", "KTPT", "CW-OGD"] + [MODEL_NAME]
#
#     if INTERPRETABLE_ANALYSIS_CONFIG['INCLUDE_ECONOMIC_DISTILLATION']:
#         PLOT_ALL_1 = PLOT_ALL_1 + [MODEL_NAME + ' (ED)']
#
#     if TUTORIAL_NAME == "TUTORIAL_1":
#         DATASET_NAME = TUTORIAL_NAME
#         DATASET_SPLIT_CONFIG = {
#             DATASET_NAME: {
#                 "TRAIN_START_TIMESTAMP": "2023-05-23",
#                 "TRAIN_END_TIMESTAMP": "2023-10-24",
#                 "VAL_START_TIMESTAMP": "2023-10-25",
#                 "VAL_END_TIMESTAMP": "2023-11-26",
#                 "TEST_START_TIMESTAMP": "2023-11-27",
#                 "TEST_END_TIMESTAMP": "2023-12-28"
#             }
#         }
