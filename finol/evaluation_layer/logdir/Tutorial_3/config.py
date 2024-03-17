import os

GET_LATEST_FINOL = True
TUTORIAL_MODE = True
TUTORIAL_NAME = "TUTORIAL_3"
ROOT_PATH = os.path.dirname(os.path.realpath(__file__))

# Parameters related to data_layer
DATASET_NAME = "SP500"  # Available options: NYSE(O), NYSE(N), DJIA, SP500, TSE, SSE, HSI, CMEG, TUTORIAL
TEST_SIZE = 0.2
VAL_SIZE = 0.25
DATASET_SPLIT_CONFIG = {
    "SP500": {
        "TRAIN_START_TIMESTAMP": "1998-01-02",
        "TRAIN_END_TIMESTAMP": "2001-03-30",
        "VAL_START_TIMESTAMP": "2001-03-31",
        "VAL_END_TIMESTAMP": "2001-10-29",
        "TEST_START_TIMESTAMP": "2001-10-30",
        "TEST_END_TIMESTAMP": "2003-01-31"
    },
    "NYSE(O)": {
        "TRAIN_START_TIMESTAMP": "1962-07-03",
        "TRAIN_END_TIMESTAMP": "1978-10-06",
        "VAL_START_TIMESTAMP": "1978-10-07",
        "VAL_END_TIMESTAMP": "1981-11-16",
        "TEST_START_TIMESTAMP": "1981-11-17",
        "TEST_END_TIMESTAMP": "1984-12-31"
    },
    "NYSE(N)": {
        "START_TIMESTAMP": "2017-01-01",
        "TEST_DURATION": 365,
        "VAL_DURATION": 180,
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
        "INCLUDE_WINDOW_DATA": False,
        "WINDOW_SIZE": 10
    }
}
NORMALIZATION_METHOD = "MIN_MAX"  # Available options: MaIN_MAX, ROBUST
BATCH_SIZE = 128
LOAD_LOCAL_DATA = False  # Set to True if you want to lod preprocessed data

# Parameters related to model_layer
MODEL_NAME = "LSRE-CAAN"
HIDDEN_SIZE = 32
NUM_LATENTS = 16
LATENT_DIM = 32
CROSS_HEADS = 1
LATENT_HEADS = 1
CROSS_DIM_HEAD = 64
LATENT_DIM_HEAD = 32
DROPOUT = 0.1

# Parameters related to optimization_layer
OPTIMIZER_NAME = "Adam"
LEARNING_RATE = 5e-4
DEVICE = "cpu"
NUM_EPOCHES = 1000

# Parameters related to evaluation_layer
BENCHMARK_BASELINE = ["Market", "Best", "UCRP", "BCRP"]
FOLLOW_THE_WINNER = ["UP", "EG", "SCRP", "PPT", "SSPO"]
FOLLOW_THE_LOSER = ["ANTI1", "ANTI2", "PAMR", "CWMR-Var", "CWMR-Stdev", "OLMAR-S", "OLMAR-E", "RMR", "RPRT"]
PATTERN_MATCHING = ["BK", "BNN", "CORN-U", "CORN-K", "AICTR", "KTPT"]
META_LEARNING = ["SP", "ONS", "GRW", "WAAS", "WAAC", "CW-OGD"]
BENCHMARK_BASELINE_ = [*BENCHMARK_BASELINE[-4:], MODEL_NAME]
FOLLOW_THE_WINNER_ = [*FOLLOW_THE_WINNER[-4:], MODEL_NAME]
FOLLOW_THE_LOSER_ = [*FOLLOW_THE_LOSER[-4:], MODEL_NAME]
PATTERN_MATCHING_ = [*PATTERN_MATCHING[-4:], MODEL_NAME]
META_LEARNING_ = [*META_LEARNING[-4:], MODEL_NAME]


################################################################################################################
############################################# FOR TUTORIAL ONLY ################################################
################################################################################################################

if TUTORIAL_MODE:
    if TUTORIAL_NAME == "TUTORIAL_1":
        DATASET_NAME = TUTORIAL_NAME
        DATASET_SPLIT_CONFIG = {
            # DATASET_NAME: {
            #     "TRAIN_START_TIMESTAMP": "2023-05-23",
            #     "TRAIN_END_TIMESTAMP": "2023-10-24",
            #     "VAL_START_TIMESTAMP": "2023-10-25",
            #     "VAL_END_TIMESTAMP": "2023-11-26",
            #     "TEST_START_TIMESTAMP": "2023-11-27",
            #     "TEST_END_TIMESTAMP": "2023-12-28"
            # }
            DATASET_NAME: {
                "TRAIN_START_TIMESTAMP": "1998-01-02",
                "TRAIN_END_TIMESTAMP": "2001-03-30",
                "VAL_START_TIMESTAMP": "2001-03-31",
                "VAL_END_TIMESTAMP": "2001-10-29",
                "TEST_START_TIMESTAMP": "2001-10-30",
                "TEST_END_TIMESTAMP": "2003-01-31"
            }
        }

    elif TUTORIAL_NAME == "TUTORIAL_2":
        DATASET_NAME = "SP500"
        DATASET_SPLIT_CONFIG = {
            "SP500": {
                "TRAIN_START_TIMESTAMP": "1998-01-02",
                "TRAIN_END_TIMESTAMP": "2001-03-30",
                "VAL_START_TIMESTAMP": "2001-03-31",
                "VAL_END_TIMESTAMP": "2001-10-29",
                "TEST_START_TIMESTAMP": "2001-10-30",
                "TEST_END_TIMESTAMP": "2003-01-31"
            }
        }

    elif TUTORIAL_NAME == "TUTORIAL_3":
        DATASET_NAME = "SP500"
        DATASET_SPLIT_CONFIG = {
            "SP500": {
                "TRAIN_START_TIMESTAMP": "1998-01-02",
                "TRAIN_END_TIMESTAMP": "2001-03-30",
                "VAL_START_TIMESTAMP": "2001-03-31",
                "VAL_END_TIMESTAMP": "2001-10-29",
                "TEST_START_TIMESTAMP": "2001-10-30",
                "TEST_END_TIMESTAMP": "2003-01-31"
            }
        }

    ################################################################################################################

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
    NORMALIZATION_METHOD = "MIN_MAX"  # Available options: MIN_MAX, ROBUST
    BATCH_SIZE = 128
    LOAD_LOCAL_DATA = False  # Set to True if you want to load preprocessed data

    # Parameters related to model_layer
    MODEL_NAME = "LSTM"
    HIDDEN_SIZE = 64
    DROPOUT = 0.1

    # Parameters related to optimization_layer
    OPTIMIZER_NAME = "Adam"
    LEARNING_RATE = 5e-4
    DEVICE = "cpu"
    NUM_EPOCHES = 200

    # Parameters related to evaluation_layer
    BENCHMARK_BASELINE = ["Market", "Best", "UCRP", "BCRP"]
    FOLLOW_THE_WINNER = ["UP", "EG", "SCRP", "PPT", "SSPO"]
    FOLLOW_THE_LOSER = ["ANTI1", "ANTI2", "PAMR", "CWMR-Var", "CWMR-Stdev", "OLMAR-S", "OLMAR-E", "RMR", "RPRT"]
    PATTERN_MATCHING = ["BK", "BNN", "CORN-U", "CORN-K", "AICTR", "KTPT"]
    META_LEARNING = ["SP", "ONS", "GRW", "WAAS", "WAAC", "CW-OGD"]
    BENCHMARK_BASELINE_ = [*BENCHMARK_BASELINE[-4:], MODEL_NAME]
    FOLLOW_THE_WINNER_ = [*FOLLOW_THE_WINNER[-4:], MODEL_NAME]
    FOLLOW_THE_LOSER_ = [*FOLLOW_THE_LOSER[-4:], MODEL_NAME]
    PATTERN_MATCHING_ = [*PATTERN_MATCHING[-4:], MODEL_NAME]
    META_LEARNING_ = [*META_LEARNING[-4:], MODEL_NAME]