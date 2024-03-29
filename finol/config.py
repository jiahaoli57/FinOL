import os
ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
PARENT_PATH = os.path.dirname(ROOT_PATH)

from finol import utils
utils.download_data()
utils.check_update(GET_LATEST_FINOL=True)

import matplotlib.pyplot as plt
plt.style.use('seaborn-talk')
from rich import print

TUTORIAL_MODE = False
TUTORIAL_NAME = "TUTORIAL_4"

# Parameters related to data_layer
DATASET_NAME = "TSE"  # Available options: NYSE(O), NYSE(N), DJIA, SP500, TSE, SSE, HSI, CMEG, CRYPTO, TUTORIAL
DATASET_SPLIT_CONFIG = {
    "NYSE(O)": {
        "TRAIN_START_TIMESTAMP": "1962-07-03",
        "TRAIN_END_TIMESTAMP": "1976-01-21",  # [3390 rows x 5 columns]
        "VAL_START_TIMESTAMP": "1976-01-22",
        "VAL_END_TIMESTAMP": "1980-07-11",  # [1130 rows x 5 columns]
        "TEST_START_TIMESTAMP": "1980-07-14",
        "TEST_END_TIMESTAMP": "1984-12-31"  # [1131 rows x 5 columns]
    },
    "NYSE(N)": {
        "TRAIN_START_TIMESTAMP": "1985-01-02",
        "TRAIN_END_TIMESTAMP": "2000-04-06",  # [3858 rows x 5 columns]
        "VAL_START_TIMESTAMP": "2000-04-07",
        "VAL_END_TIMESTAMP": "2005-05-20",  # [1286 rows x 5 columns]
        "TEST_START_TIMESTAMP": "2005-05-23",
        "TEST_END_TIMESTAMP": "2010-06-30"  # [1286 rows x 5 columns]
    },
    "DJIA": {
        "TRAIN_START_TIMESTAMP": "2001-01-14",
        "TRAIN_END_TIMESTAMP": "2002-04-01",  # [300 rows x 5 columns]
        "VAL_START_TIMESTAMP": "2002-04-02",
        "VAL_END_TIMESTAMP": "2002-08-21",  # [100 rows x 5 columns]
        "TEST_START_TIMESTAMP": "2002-08-22",
        "TEST_END_TIMESTAMP": "2003-01-14"  # [100 rows x 5 columns]
    },
    "SP500": {
        "TRAIN_START_TIMESTAMP": "1998-01-02",
        "TRAIN_END_TIMESTAMP": "2001-01-12",  # [756 rows x 5 columns]
        "VAL_START_TIMESTAMP": "2001-01-16",
        "VAL_END_TIMESTAMP": "2002-01-25",  # [256 rows x 5 columns]
        "TEST_START_TIMESTAMP": "2002-01-28",
        "TEST_END_TIMESTAMP": "2003-01-31"  # [256 rows x 5 columns]
    },
    "TSE": {
        "TRAIN_START_TIMESTAMP": "1995-01-12",
        "TRAIN_END_TIMESTAMP": "1997-05-28",  # [600 rows x 5 columns] NUM OF PERIODS IN TRAINING SETS ARE UNCERATIN
        "VAL_START_TIMESTAMP": "1997-05-29",
        "VAL_END_TIMESTAMP": "1998-03-13",  # [200 rows x 5 columns]
        "TEST_START_TIMESTAMP": "1998-03-16",
        "TEST_END_TIMESTAMP": "1998-12-31"  # [201 rows x 5 columns]
    },
    "SSE": {
        "TRAIN_START_TIMESTAMP": "2010-07-05",
        "TRAIN_END_TIMESTAMP": "2018-04-09",  # [406 rows x 5 columns]
        "VAL_START_TIMESTAMP": "2018-04-16",
        "VAL_END_TIMESTAMP": "2020-11-16",  # [136 rows x 5 columns]
        "TEST_START_TIMESTAMP": "2020-11-23",
        "TEST_END_TIMESTAMP": "2023-06-26"  # [136 rows x 5 columns]
    },
    "HSI": {
        "TRAIN_START_TIMESTAMP": "2010-07-05",
        "TRAIN_END_TIMESTAMP": "2018-04-09",  # [406 rows x 5 columns]
        "VAL_START_TIMESTAMP": "2018-04-16",
        "VAL_END_TIMESTAMP": "2020-11-16",  # [136 rows x 5 columns]
        "TEST_START_TIMESTAMP": "2020-11-23",
        "TEST_END_TIMESTAMP": "2023-06-26"  # [136 rows x 5 columns]
    },
    "CMEG": {
        "TRAIN_START_TIMESTAMP": "2010-07-05",
        "TRAIN_END_TIMESTAMP": "2018-04-09",  # [406 rows x 5 columns]
        "VAL_START_TIMESTAMP": "2018-04-16",
        "VAL_END_TIMESTAMP": "2020-11-16",  # [136 rows x 5 columns]
        "TEST_START_TIMESTAMP": "2020-11-23",
        "TEST_END_TIMESTAMP": "2023-06-26"  # [136 rows x 5 columns]
    },
    "CRYPTO": {
        "TRAIN_START_TIMESTAMP": "2017-11-09",
        "TRAIN_END_TIMESTAMP": "2021-08-22",  # [1383 rows x 5 columns]
        "VAL_START_TIMESTAMP": "2021-08-23",
        "VAL_END_TIMESTAMP": "2022-11-26",  # [461 rows x 5 columns]
        "TEST_START_TIMESTAMP": "2022-11-27",
        "TEST_END_TIMESTAMP": "2024-03-01"  # [461 rows x 5 columns]
    }
}
FEATURE_ENGINEERING_CONFIG = {
    "INCLUDE_OHLCV_FEATURES": True,
    "INCLUDE_OVERLAP_FEATURES": True,
    "INCLUDE_MOMENTUM_FEATURES": True,
    "INCLUDE_VOLUME_FEATURES": True,
    "INCLUDE_CYCLE_FEATURES": True,
    "INCLUDE_PRICE_FEATURES": True,
    "INCLUDE_VOLATILITY_FEATURES": True,
    "INCLUDE_PATTERN_FEATURES": True
}
DATA_AUGMENTATION_CONFIG = {
    "WINDOW_DATA": {
        "INCLUDE_WINDOW_DATA": True,
        "WINDOW_SIZE": 10
    }
}
SCALER = "MinMaxScaler"  # None, StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
BATCH_SIZE = {
    "NYSE(O)": 128,
    "NYSE(N)": 128,
    "DJIA": 32,
    "SP500": 64,
    "TSE": 64,
    "SSE": 32,
    "HSI": 32,
    "CMEG": 32,
    "CRYPTO": 128,
}

# Parameters related to model_layer
MODEL_NAME = "DNN"
MODEL_CONFIG = {
    "DNN": {
        "NUM_LAYERS": 1,
        "HIDDEN_SIZE": 32
    },
    "RNN": {
        "NUM_LAYERS": 1,
        "HIDDEN_SIZE": 32
    },
    "LSTM": {
        "NUM_LAYERS": 1,
        "HIDDEN_SIZE": 32
    },
    "CNN": {
        "OUT_CHANNELS": 128,
        "KERNEL_SIZE": 3,
        "STRIDE": 1,
        "HIDDEN_SIZE": 32,
    },
    "Transformer": {
        "NUM_LAYERS": 1,
        "NUM_HEADS": 1,
        "HIDDEN_SIZE": 32,
    },
    "LSRE-CAAN": {
        "NUM_LATENTS": 16,
        "LATENT_DIM": 32,
        "CROSS_HEADS": 1,
        "LATENT_HEADS": 1,
        "CROSS_DIM_HEAD": 64,
        "LATENT_DIM_HEAD": 32,
        "HIDDEN_SIZE": 32,
    },
}
DROPOUT = 0.1

# Parameters related to optimization_layer
MANUAL_SEED = 42
OPTIMIZER_NAME = "Adam"
LEARNING_RATE = 1e-4
CRITERION_NAME = "LOG_SINGLE_PERIOD_WEALTH"
LAMBDA_L2 = 0.001
DEVICE = "cpu"
NUM_EPOCHES = 1000
from finol.optimization_layer import criterion_selector

# Parameters related to evaluation_layer
MARKERS = ['o', '^', '<', '>', 's', 'p', 'h', '+', 'x', '|', '_']
MARKEVERY = 10
ALPHA = 0.5
METRIC_CONFIG = {
    "INCLUDE_PROFIT_METRICS": True,
    "INCLUDE_RISK_METRICS": True,
    "PRACTICAL_METRICS": {
        "INCLUDE_PRACTICAL_METRICS": True,
        "TRANSACTIOS_COSTS_RATE": 1/100,
        "TRANSACTIOS_COSTS_RATE_INTERVAL": 1/1000,
    }
}
BENCHMARK_BASELINE = ["Market", "Best", "UCRP", "BCRP"]
FOLLOW_THE_WINNER = ["UP", "EG", "SCRP", "PPT", "SSPO"]
FOLLOW_THE_LOSER = ["ANTI1", "ANTI2", "PAMR", "CWMR-Var", "CWMR-Stdev", "OLMAR-S", "OLMAR-E", "RMR", "RPRT"]
PATTERN_MATCHING = ["AICTR", "KTPT"]
META_LEARNING = ["SP", "ONS", "GRW", "WAAS", "CW-OGD"]

# PLOT_ALL_1 = ["EG", "PPT", "RMR", "RPRT", "Best"]
# PLOT_ALL_2 = ["AICTR", "KTPT", "SP", "CW-OGD", "Best"]

PLOT_ALL_1 = BENCHMARK_BASELINE + [MODEL_NAME]
PLOT_ALL_2 = FOLLOW_THE_WINNER + [MODEL_NAME]
PLOT_ALL_3 = FOLLOW_THE_LOSER + [MODEL_NAME]
PLOT_ALL_4 = PATTERN_MATCHING + [MODEL_NAME]
PLOT_ALL_5 = META_LEARNING + [MODEL_NAME]



################################################################################################################
############################################# FOR TUTORIAL ONLY ################################################
################################################################################################################

if TUTORIAL_MODE:
    DATASET_NAME = "NYSE(O)"
    DATASET_SPLIT_CONFIG = {
        "NYSE(O)": {
            "TRAIN_START_TIMESTAMP": "1962-07-03",
            "TRAIN_END_TIMESTAMP": "1976-01-21",  # [3390 rows x 5 columns]
            "VAL_START_TIMESTAMP": "1976-01-22",
            "VAL_END_TIMESTAMP": "1980-07-11",  # [1130 rows x 5 columns]
            "TEST_START_TIMESTAMP": "1980-07-14",
            "TEST_END_TIMESTAMP": "1984-12-31"  # [1131 rows x 5 columns]
        },
        "NYSE(N)": {
            "TRAIN_START_TIMESTAMP": "1985-01-02",
            "TRAIN_END_TIMESTAMP": "2000-04-06",  # [3858 rows x 5 columns]
            "VAL_START_TIMESTAMP": "2000-04-07",
            "VAL_END_TIMESTAMP": "2005-05-20",  # [1286 rows x 5 columns]
            "TEST_START_TIMESTAMP": "2005-05-23",
            "TEST_END_TIMESTAMP": "2010-06-30"  # [1286 rows x 5 columns]
        },
        "DJIA": {
            "TRAIN_START_TIMESTAMP": "2001-01-14",
            "TRAIN_END_TIMESTAMP": "2002-04-01",  # [300 rows x 5 columns]
            "VAL_START_TIMESTAMP": "2002-04-02",
            "VAL_END_TIMESTAMP": "2002-08-21",  # [100 rows x 5 columns]
            "TEST_START_TIMESTAMP": "2002-08-22",
            "TEST_END_TIMESTAMP": "2003-01-14"  # [100 rows x 5 columns]
        },
        "SP500": {
            "TRAIN_START_TIMESTAMP": "1998-01-02",
            "TRAIN_END_TIMESTAMP": "2001-01-12",  # [756 rows x 5 columns]
            "VAL_START_TIMESTAMP": "2001-01-16",
            "VAL_END_TIMESTAMP": "2002-01-25",  # [256 rows x 5 columns]
            "TEST_START_TIMESTAMP": "2002-01-28",
            "TEST_END_TIMESTAMP": "2003-01-31"  # [256 rows x 5 columns]
        },
        "TSE": {
            "TRAIN_START_TIMESTAMP": "1995-01-12",
            "TRAIN_END_TIMESTAMP": "1997-05-28",  # [600 rows x 5 columns] NUM OF PERIODS IN TRAINING SETS ARE UNCERATIN
            "VAL_START_TIMESTAMP": "1997-05-29",
            "VAL_END_TIMESTAMP": "1998-03-13",  # [200 rows x 5 columns]
            "TEST_START_TIMESTAMP": "1998-03-16",
            "TEST_END_TIMESTAMP": "1998-12-31"  # [201 rows x 5 columns]
        },
        "SSE": {
            "TRAIN_START_TIMESTAMP": "2010-07-05",
            "TRAIN_END_TIMESTAMP": "2018-04-09",  # [406 rows x 5 columns]
            "VAL_START_TIMESTAMP": "2018-04-16",
            "VAL_END_TIMESTAMP": "2020-11-16",  # [136 rows x 5 columns]
            "TEST_START_TIMESTAMP": "2020-11-23",
            "TEST_END_TIMESTAMP": "2023-06-26"  # [136 rows x 5 columns]
        },
        "HSI": {
            "TRAIN_START_TIMESTAMP": "2010-07-05",
            "TRAIN_END_TIMESTAMP": "2018-04-09",  # [406 rows x 5 columns]
            "VAL_START_TIMESTAMP": "2018-04-16",
            "VAL_END_TIMESTAMP": "2020-11-16",  # [136 rows x 5 columns]
            "TEST_START_TIMESTAMP": "2020-11-23",
            "TEST_END_TIMESTAMP": "2023-06-26"  # [136 rows x 5 columns]
        },
        "CMEG": {
            "TRAIN_START_TIMESTAMP": "2010-07-05",
            "TRAIN_END_TIMESTAMP": "2018-04-09",  # [406 rows x 5 columns]
            "VAL_START_TIMESTAMP": "2018-04-16",
            "VAL_END_TIMESTAMP": "2020-11-16",  # [136 rows x 5 columns]
            "TEST_START_TIMESTAMP": "2020-11-23",
            "TEST_END_TIMESTAMP": "2023-06-26"  # [136 rows x 5 columns]
        },
        "CRYPTO": {
            "TRAIN_START_TIMESTAMP": "2017-11-09",
            "TRAIN_END_TIMESTAMP": "2021-08-22",  # [1383 rows x 5 columns]
            "VAL_START_TIMESTAMP": "2021-08-23",
            "VAL_END_TIMESTAMP": "2022-11-26",  # [461 rows x 5 columns]
            "TEST_START_TIMESTAMP": "2022-11-27",
            "TEST_END_TIMESTAMP": "2024-03-01"  # [461 rows x 5 columns]
        },
    }
    FEATURE_ENGINEERING_CONFIG = {
        "INCLUDE_OHLCV_FEATURES": True,
        "INCLUDE_OVERLAP_FEATURES": True,
        "INCLUDE_MOMENTUM_FEATURES": True,
        "INCLUDE_VOLUME_FEATURES": True,
        "INCLUDE_CYCLE_FEATURES": True,
        "INCLUDE_PRICE_FEATURES": True,
        "INCLUDE_VOLATILITY_FEATURES": True,
        "INCLUDE_PATTERN_FEATURES": True
    }
    DATA_AUGMENTATION_CONFIG = {
        "WINDOW_DATA": {
            "INCLUDE_WINDOW_DATA": True,
            "WINDOW_SIZE": 10
        }
    }
    SCALER = "MinMaxScaler"  # None, StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
    BATCH_SIZE = {
        "NYSE(O)": 128,
        "NYSE(N)": 128,
        "DJIA": 32,
        "SP500": 64,
        "TSE": 64,
        "SSE": 32,
        "HSI": 32,
        "CMEG": 32,
        "CRYPTO": 128,
    }

    # Parameters related to model_layer
    MODEL_NAME = "LSTM"
    MODEL_CONFIG = {
        "DNN": {
            "NUM_LAYERS": 1,
            "HIDDEN_SIZE": 32
        },
        "RNN": {
            "NUM_LAYERS": 1,
            "HIDDEN_SIZE": 32
        },
        "LSTM": {
            "NUM_LAYERS": 1,
            "HIDDEN_SIZE": 32
        },
        "CNN": {
            "OUT_CHANNELS": 128,
            "KERNEL_SIZE": 3,
            "STRIDE": 1,
            "HIDDEN_SIZE": 32,
        },
        "Transformer": {
            "NUM_LAYERS": 1,
            "NUM_HEADS": 1,
            "HIDDEN_SIZE": 32,
        },
        "LSRE-CAAN": {
            "NUM_LATENTS": 16,
            "LATENT_DIM": 32,
            "CROSS_HEADS": 1,
            "LATENT_HEADS": 1,
            "CROSS_DIM_HEAD": 64,
            "LATENT_DIM_HEAD": 32,
            "HIDDEN_SIZE": 32,
        },
    }
    DROPOUT = 0.1

    # Parameters related to optimization_layer
    MANUAL_SEED = 42
    OPTIMIZER_NAME = "Adam"
    LEARNING_RATE = 1e-4
    CRITERION_NAME = "LOG_SINGLE_PERIOD_WEALTH"
    LAMBDA_L2 = 0.001
    DEVICE = "cpu"
    NUM_EPOCHES = 100

    # Parameters related to evaluation_layer
    MARKERS = ['o', '^', '<', '>', 's', 'p', 'h', '+', 'x', '|', '_']
    MARKEVERY = 10
    ALPHA = 0.5
    METRIC_CONFIG = {
        "INCLUDE_PROFIT_METRICS": True,
        "INCLUDE_RISK_METRICS": True,
        "PRACTICAL_METRICS": {
            "INCLUDE_PRACTICAL_METRICS": True,
            "TRANSACTIOS_COSTS_RATE": 1 / 100,
            "TRANSACTIOS_COSTS_RATE_INTERVAL": 1 / 1000,
        }
    }

    if TUTORIAL_NAME == "TUTORIAL_1":
        DATASET_NAME = TUTORIAL_NAME
        DATASET_SPLIT_CONFIG = {
            DATASET_NAME: {
                "TRAIN_START_TIMESTAMP": "2023-05-23",
                "TRAIN_END_TIMESTAMP": "2023-10-24",
                "VAL_START_TIMESTAMP": "2023-10-25",
                "VAL_END_TIMESTAMP": "2023-11-26",
                "TEST_START_TIMESTAMP": "2023-11-27",
                "TEST_END_TIMESTAMP": "2023-12-28"
            }
        }

