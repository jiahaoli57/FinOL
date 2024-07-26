import os
import time
import torch
import pandas as pd

from tqdm import tqdm
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from finol.data_layer.ScalerSelector import ScalerSelector
from finol.utils import ROOT_PATH, load_config, update_config, make_logdir, check_update, download_data


class DatasetLoader:
    """
    DatasetLoader is a class that provides methods to load various datasets.
    """
    def __init__(self):
        self.config = load_config()
        check_update()
        download_data()

    def data_accessing(self, folder_path: str) -> list[pd.DataFrame]:
        """

        :param folder_path:
        :return: raw_files
        """
        raw_files = []
        for file_name in tqdm(os.listdir(folder_path), desc="Data Loading"):
            if file_name.endswith(".xlsx") or file_name.endswith(".xls"):
                file_path = os.path.join(folder_path, file_name)
                try:
                    dataframe = pd.read_excel(file_path)
                    raw_files.append(dataframe)
                except Exception as e:
                    print(f"An error occurred while loading file {file_path}: {str(e)}")
        return raw_files

    def feature_engineering(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
        """
        Performs feature engineering on the input DataFrame to generate various types of features.

        :param df: The input DataFrame containing the raw data.
        :return: A tuple containing the processed DataFrame, a list of the detailed feature names, and a dictionary
        containing the number of features for each category.
        """
        ohlcv_features_df = pd.DataFrame()
        overlap_features_df = pd.DataFrame()
        momentum_features_df = pd.DataFrame()
        volume_features_df = pd.DataFrame()
        cycle_features_df = pd.DataFrame()
        price_features_df = pd.DataFrame()
        volatility_features_df = pd.DataFrame()
        pattern_features_df = pd.DataFrame()

        if self.config["FEATURE_ENGINEERING_CONFIG"]["INCLUDE_OHLCV_FEATURES"]:
            ohlcv_features = {
                "OPEN": df.OPEN,
                "HIGH": df.HIGH,
                "LOW": df.LOW,
                "CLOSE": df.CLOSE,
                "VOLUME": df.VOLUME
            }
            ohlcv_features_df = pd.DataFrame(ohlcv_features)

        if self.config["FEATURE_ENGINEERING_CONFIG"]["INCLUDE_OVERLAP_FEATURES"]:
            import talib as ta
            overlap_features = {
                "BBANDS_UPPER": ta.BBANDS(df.CLOSE)[0],  # Bollinger Bands - Upper Band
                "BBANDS_MIDDLE": ta.BBANDS(df.CLOSE)[1],  # Bollinger Bands - Middle Band
                "BBANDS_LOWER": ta.BBANDS(df.CLOSE)[2],  # Bollinger Bands - Lower Band
                "DEMA": ta.DEMA(df.CLOSE),  # Double Exponential Moving Average
                "EMA": ta.EMA(df.CLOSE),  # Exponential Moving Average
                "HT_TRENDLINE": ta.HT_TRENDLINE(df.CLOSE),  # Hilbert Transform - Instantaneous Trendline
                "KAMA": ta.KAMA(df.CLOSE),  # Kaufman Adaptive Moving Average
                "MA": ta.MA(df.CLOSE),  # Moving Average
                "MAMA": ta.MAMA(df.CLOSE)[0],  # MESA Adaptive Moving Average - MAMA
                "MAMA_FAMA": ta.MAMA(df.CLOSE)[1],  # MESA Adaptive Moving Average - FAMA
                "MAVP": ta.MAVP(df.CLOSE, df.DATE),  # Moving Average with Variable Period
                "MIDPOINT": ta.MIDPOINT(df.CLOSE),  # MidPoint over Period
                "MIDPRICE": ta.MIDPRICE(df.HIGH, df.LOW),  # Midpoint Price over Period
                "SAR": ta.SAR(df.HIGH, df.LOW),  # Parabolic SAR
                "SAREXT": ta.SAREXT(df.HIGH, df.LOW),  # Parabolic SAR - Extended
                "SMA": ta.SMA(df.CLOSE),  # Simple Moving Average
                "T3": ta.T3(df.CLOSE),  # Triple Exponential Moving Average (T3)
                "TEMA": ta.TEMA(df.CLOSE),  # Triple Exponential Moving Average
                "TRIMA": ta.TRIMA(df.CLOSE),  # Triangular Moving Average
                "WMA": ta.WMA(df.CLOSE)  # Weighted Moving Average
            }
            overlap_features_df = pd.DataFrame(overlap_features)

        if self.config["FEATURE_ENGINEERING_CONFIG"]["INCLUDE_MOMENTUM_FEATURES"]:
            import talib as ta
            momentum_features = {
                "ADX": ta.ADX(df.HIGH, df.LOW, df.CLOSE),  # Average Directional Movement Index
                "ADXR": ta.ADXR(df.HIGH, df.LOW, df.CLOSE),  # Average Directional Movement Index Rating
                "APO": ta.APO(df.CLOSE),  # Absolute Price Oscillator
                "AROON_UP": ta.AROON(df.HIGH, df.LOW)[0],  # Aroon Up
                "AROON_DOWN": ta.AROON(df.HIGH, df.LOW)[1],  # Aroon Down
                "AROONOSC": ta.AROONOSC(df.HIGH, df.LOW),  # Aroon Oscillator
                "BOP": ta.BOP(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Balance Of Power
                "CCI": ta.CCI(df.HIGH, df.LOW, df.CLOSE),  # Commodity Channel Index
                "CMO": ta.CMO(df.CLOSE),  # Chande Momentum Oscillator
                "DX": ta.DX(df.HIGH, df.LOW, df.CLOSE),  # Directional Movement Index
                "MACD": ta.MACD(df.CLOSE)[0],  # Moving Average Convergence/Divergence
                "MACD_SIGNAL": ta.MACD(df.CLOSE)[1],  # MACD Signal Line
                "MACD_HIST": ta.MACD(df.CLOSE)[2],  # MACD Histogram
                "MACDEXT": ta.MACDEXT(df.CLOSE)[0],  # MACD with controllable MA type
                "MACDEXT_SIGNAL": ta.MACDEXT(df.CLOSE)[1],  # MACDEXT Signal Line
                "MACDEXT_HIST": ta.MACDEXT(df.CLOSE)[2],  # MACDEXT Histogram
                "MACDFIX": ta.MACDFIX(df.CLOSE)[0],  # Moving Average Convergence/Divergence Fix 12/26
                "MACDFIX_SIGNAL": ta.MACDFIX(df.CLOSE)[1],  # MACDFIX Signal Line
                "MACDFIX_HIST": ta.MACDFIX(df.CLOSE)[2],  # MACDFIX Histogram
                "MFI": ta.MFI(df.HIGH, df.LOW, df.CLOSE, df.VOLUME),  # Money Flow Index
                "MINUS_DI": ta.MINUS_DI(df.HIGH, df.LOW, df.CLOSE),  # Minus Directional Indicator
                "MINUS_DM": ta.MINUS_DM(df.HIGH, df.LOW),  # Minus Directional Movement
                "MOM": ta.MOM(df.CLOSE),  # Momentum
                "PLUS_DI": ta.PLUS_DI(df.HIGH, df.LOW, df.CLOSE),  # Plus Directional Indicator
                "PLUS_DM": ta.PLUS_DM(df.HIGH, df.LOW),  # Plus Directional Movement
                "PPO": ta.PPO(df.CLOSE),  # Percentage Price Oscillator
                "ROC": ta.ROC(df.CLOSE),  # Rate of change: ((price/prevPrice)-1)*100
                "ROCP": ta.ROCP(df.CLOSE),  # Rate of change Percentage: (price-prevPrice)/prevPrice
                "ROCR": ta.ROCR(df.CLOSE),  # Rate of change ratio: (price/prevPrice)
                "ROCR100": ta.ROCR100(df.CLOSE),  # Rate of change ratio 100 scale: (price/prevPrice)*100
                "RSI": ta.RSI(df.CLOSE),  # Relative Strength Index
                "STOCH_K": ta.STOCH(df.HIGH, df.LOW, df.CLOSE)[0],  # Stochastic %K
                "STOCH_D": ta.STOCH(df.HIGH, df.LOW, df.CLOSE)[1],  # Stochastic %D
                "STOCHF_K": ta.STOCHF(df.HIGH, df.LOW, df.CLOSE)[0],  # Stochastic Fast %K
                "STOCHF_D": ta.STOCHF(df.HIGH, df.LOW, df.CLOSE)[1],  # Stochastic Fast %D
                "STOCHRSI_K": ta.STOCHRSI(df.CLOSE)[0],  # Stochastic RSI %K
                "STOCHRSI_D": ta.STOCHRSI(df.CLOSE)[1],  # Stochastic RSI %D
                "TRIX": ta.TRIX(df.CLOSE),  # 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA
                "ULTOSC": ta.ULTOSC(df.HIGH, df.LOW, df.CLOSE),  # Ultimate Oscillator
                "WILLR": ta.WILLR(df.HIGH, df.LOW, df.CLOSE)  # Williams" %R
            }
            momentum_features_df = pd.DataFrame(momentum_features)

        if self.config["FEATURE_ENGINEERING_CONFIG"]["INCLUDE_VOLUME_FEATURES"]:
            import talib as ta
            volume_features = {
                "AD": ta.AD(df.HIGH, df.LOW, df.CLOSE, df.VOLUME),  # Chaikin A/D Line
                "ADOSC": ta.ADOSC(df.HIGH, df.LOW, df.CLOSE, df.VOLUME),  # Chaikin A/D Oscillator
                "OBV": ta.OBV(df.CLOSE, df.VOLUME)  # On Balance Volume
            }
            volume_features_df = pd.DataFrame(volume_features)

        if self.config["FEATURE_ENGINEERING_CONFIG"]["INCLUDE_CYCLE_FEATURES"]:
            import talib as ta
            cycle_features = {
                "HT_DCPERIOD": ta.HT_DCPERIOD(df.CLOSE),  # Hilbert Transform - Dominant Cycle Period
                "HT_DCPHASE": ta.HT_DCPHASE(df.CLOSE),  # Hilbert Transform - Dominant Cycle Phase
                "HT_PHASOR_INPHASE": ta.HT_PHASOR(df.CLOSE)[0],  # Hilbert Transform - Phasor Components
                "HT_PHASOR_QUADRATURE": ta.HT_PHASOR(df.CLOSE)[1],  # Hilbert Transform - Phasor Components
                "HT_SINE_LEADSINE": ta.HT_SINE(df.CLOSE)[0],  # Hilbert Transform - SineWave
                "HT_SINE_SINEWAVE": ta.HT_SINE(df.CLOSE)[1],  # Hilbert Transform - SineWave
                "HT_TRENDMODE": ta.HT_TRENDMODE(df.CLOSE)  # Hilbert Transform - Trend vs Cycle Mode
            }
            cycle_features_df = pd.DataFrame(cycle_features)

        if self.config["FEATURE_ENGINEERING_CONFIG"]["INCLUDE_PRICE_FEATURES"]:
            import talib as ta
            price_features = {
                "AVGPRICE": ta.AVGPRICE(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Average Price
                "MEDPRICE": ta.MEDPRICE(df.HIGH, df.LOW),  # Median Price
                "TYPPRICE": ta.TYPPRICE(df.HIGH, df.LOW, df.CLOSE),  # Typical Price
                "WCLPRICE": ta.WCLPRICE(df.HIGH, df.LOW, df.CLOSE)  # Weighted Close Price
            }
            price_features_df = pd.DataFrame(price_features)

        if self.config["FEATURE_ENGINEERING_CONFIG"]["INCLUDE_VOLATILITY_FEATURES"]:
            import talib as ta
            volatility_features = {
                "ATR": ta.ATR(df.HIGH, df.LOW, df.CLOSE),  # Average True Range
                "NATR": ta.NATR(df.HIGH, df.LOW, df.CLOSE),  # Normalized Average True Range
                "TRANGE": ta.TRANGE(df.HIGH, df.LOW, df.CLOSE)  # True Range
            }
            volatility_features_df = pd.DataFrame(volatility_features)

        if self.config["FEATURE_ENGINEERING_CONFIG"]["INCLUDE_PATTERN_FEATURES"]:
            import talib as ta
            pattern_features = {
                "CDL2CROWS": ta.CDL2CROWS(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Two Crows
                "CDL3BLACKCROWS": ta.CDL3BLACKCROWS(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Three Black Crows
                "CDL3INSIDE": ta.CDL3INSIDE(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Three Inside Up/Down
                "CDL3LINESTRIKE": ta.CDL3LINESTRIKE(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Three-Line Strike
                "CDL3OUTSIDE": ta.CDL3OUTSIDE(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Three Outside Up/Down
                "CDL3STARSINSOUTH": ta.CDL3STARSINSOUTH(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Three Stars In The South
                "CDL3WHITESOLDIERS": ta.CDL3WHITESOLDIERS(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Three Advancing White Soldiers
                "CDLABANDONEDBABY": ta.CDLABANDONEDBABY(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Abandoned Baby
                "CDLADVANCEBLOCK": ta.CDLADVANCEBLOCK(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Advance Block
                "CDLBELTHOLD": ta.CDLBELTHOLD(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Belt-Hold
                "CDLBREAKAWAY": ta.CDLBREAKAWAY(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Breakaway
                "CDLCLOSINGMARUBOZU": ta.CDLCLOSINGMARUBOZU(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Closing Marubozu
                "CDLCONCEALBABYSWALL": ta.CDLCONCEALBABYSWALL(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Concealing Baby Swallow
                "CDLCOUNTERATTACK": ta.CDLCOUNTERATTACK(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Counterattack
                "CDLDARKCLOUDCOVER": ta.CDLDARKCLOUDCOVER(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Dark Cloud Cover
                "CDLDOJI": ta.CDLDOJI(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Doji,
                "CDLDOJISTAR": ta.CDLDOJISTAR(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Doji Star
                "CDLDRAGONFLYDOJI": ta.CDLDRAGONFLYDOJI(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Dragonfly Doji
                "CDLENGULFING": ta.CDLENGULFING(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Engulfing Pattern
                "CDLEVENINGDOJISTAR": ta.CDLEVENINGDOJISTAR(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Evening Doji Star
                "CDLEVENINGSTAR": ta.CDLEVENINGSTAR(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Evening Star
                "CDLGAPSIDESIDEWHITE": ta.CDLGAPSIDESIDEWHITE(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Up/Down-Gap Side-By-Side White Lines
                "CDLGRAVESTONEDOJI": ta.CDLGRAVESTONEDOJI(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Gravestone Doji
                "CDLHAMMER": ta.CDLHAMMER(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Hammer
                "CDLHANGINGMAN": ta.CDLHANGINGMAN(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Hanging Man
                "CDLHARAMI": ta.CDLHARAMI(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Harami Pattern
                "CDLHARAMICROSS": ta.CDLHARAMICROSS(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Harami Cross Pattern
                "CDLHIGHWAVE": ta.CDLHIGHWAVE(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # High-Wave Candle
                "CDLHIKKAKE": ta.CDLHIKKAKE(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Hikkake Pattern
                "CDLHIKKAKEMOD": ta.CDLHIKKAKEMOD(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Modified Hikkake Pattern
                "CDLHOMINGPIGEON": ta.CDLHOMINGPIGEON(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Homing Pigeon
                "CDLIDENTICAL3CROWS": ta.CDLIDENTICAL3CROWS(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Identical Three Crows
                "CDLINNECK": ta.CDLINNECK(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # In-Neck Pattern
                "CDLINVERTEDHAMMER": ta.CDLINVERTEDHAMMER(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Inverted Hammer
                "CDLKICKING": ta.CDLKICKING(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Kicking
                "CDLKICKINGBYLENGTH": ta.CDLKICKINGBYLENGTH(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Kicking - Bull/Bear Determined by the Longer Marubozu
                "CDLLADDERBOTTOM": ta.CDLLADDERBOTTOM(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Ladder Bottom
                "CDLLONGLEGGEDDOJI": ta.CDLLONGLEGGEDDOJI(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Long Legged Doji
                "CDLLONGLINE": ta.CDLLONGLINE(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Long Line Candle
                "CDLMARUBOZU": ta.CDLMARUBOZU(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Marubozu
                "CDLMATCHINGLOW": ta.CDLMATCHINGLOW(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Matching Low
                "CDLMATHOLD": ta.CDLMATHOLD(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Mat Hold
                "CDLMORNINGDOJISTAR": ta.CDLMORNINGDOJISTAR(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Morning Doji Star
                "CDLMORNINGSTAR": ta.CDLMORNINGSTAR(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Morning Star
                "CDLONNECK": ta.CDLONNECK(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # On-Neck Pattern
                "CDLPIERCING": ta.CDLPIERCING(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Piercing Pattern
                "CDLRICKSHAWMAN": ta.CDLRICKSHAWMAN(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Rickshaw Man
                "CDLRISEFALL3METHODS": ta.CDLRISEFALL3METHODS(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Rising/Falling Three Methods
                "CDLSEPARATINGLINES": ta.CDLSEPARATINGLINES(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Separating Lines
                "CDLSHOOTINGSTAR": ta.CDLSHOOTINGSTAR(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Shooting Star
                "CDLSHORTLINE": ta.CDLSHORTLINE(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Short Line Candle
                "CDLSPINNINGTOP": ta.CDLSPINNINGTOP(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Spinning Top
                "CDLSTALLEDPATTERN": ta.CDLSTALLEDPATTERN(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Stalled Pattern
                "CDLSTICKSANDWICH": ta.CDLSTICKSANDWICH(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Stick Sandwich
                "CDLTAKURI": ta.CDLTAKURI(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Takuri (Dragonfly Doji with Very Long Lower Shadow)
                "CDLTASUKIGAP": ta.CDLTASUKIGAP(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Tasuki Gap
                "CDLTHRUSTING": ta.CDLTHRUSTING(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Thrusting Pattern
                "CDLTRISTAR": ta.CDLTRISTAR(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Tristar Pattern
                "CDLUNIQUE3RIVER": ta.CDLUNIQUE3RIVER(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Unique 3 River
                "CDLUPSIDEGAP2CROWS": ta.CDLUPSIDEGAP2CROWS(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Upside Gap Two Crows
                "CDLXSIDEGAP3METHODS": ta.CDLXSIDEGAP3METHODS(df.OPEN, df.HIGH, df.LOW, df.CLOSE)  # Upside/Downside Gap Three Methods
            }
            pattern_features_df = pd.DataFrame(pattern_features)

        _ = pd.concat([df["DATE"], ohlcv_features_df, overlap_features_df, momentum_features_df, volume_features_df, cycle_features_df,
                        price_features_df, volatility_features_df, pattern_features_df], axis=1)
        _.set_index("DATE", inplace=True)
        DETAILED_FEATURE_LIST = _.columns.tolist()
        DETAILED_NUM_FEATURES = {
            "OHLCV_FEATURES": ohlcv_features_df.shape[1],
            "OVERLAP_FEATURES": overlap_features_df.shape[1],
            "MOMENTUM_FEATURES": momentum_features_df.shape[1],
            "VOLUME_FEATURES": volume_features_df.shape[1],
            "CYCLE_FEATURES": cycle_features_df.shape[1],
            "PRICE_FEATURES": price_features_df.shape[1],
            "VOLATILITY_FEATURES": volatility_features_df.shape[1],
            "PATTERN_FEATURES": pattern_features_df.shape[1],
        }
        print(DETAILED_FEATURE_LIST)
        print(DETAILED_NUM_FEATURES)
        return _, DETAILED_FEATURE_LIST, DETAILED_NUM_FEATURES

    def data_augmentation(self, df):
        if self.config["DATA_AUGMENTATION_CONFIG"]["WINDOW_DATA"]["INCLUDE_WINDOW_DATA"]:
            WINDOW_SIZE = self.config["DATA_AUGMENTATION_CONFIG"]["WINDOW_DATA"]["WINDOW_SIZE"]
            df = pd.concat([df] + [df.shift(i).add_prefix(f"prev_{i}_") for i in range(1, WINDOW_SIZE)], axis=1)
        else:
            WINDOW_SIZE = 1
        return df, WINDOW_SIZE

    def data_cleaning(self, df):
        """
        Clean the DataFrame by removing rows with missing values.
        """
        return df.dropna(how="any")

    def zscore_calculation(self, df):
        df_normalized = df.copy()
        numeric_features = df.select_dtypes(include=["int", "float"]).columns
        scaler = ScalerSelector().select_scaler()
        if scaler != None:
            zscore = scaler.fit(df_normalized[numeric_features])  # zscore is different for different assets
        else:
            zscore = None
        return zscore

    def data_normalization(self, df, zscore):
        """
        Normalize all numeric features in DataFrame
        """
        df_normalized = df.copy()
        numeric_features = df.select_dtypes(include=["int", "float"]).columns
        if zscore != None:
            df_normalized[numeric_features] = zscore.transform(df_normalized[numeric_features])
            # print(df_normalized[numeric_features])
            # print(df_normalized[numeric_features]["OPEN"].values)
            # print(df_normalized[numeric_features]["OPEN"].values.mean())  # -3.620599927440001e-16
            # print(df_normalized[numeric_features]["OPEN"].values.std())  # 1
            # time.sleep(1111)
        else:
            df_normalized[numeric_features] = df_normalized[numeric_features]
        return df_normalized

    def data_splitting(self, df):
        """
        Split the DataFrame into train, validation, and test sets.
        """
        train_start = pd.to_datetime(self.config["DATASET_SPLIT_CONFIG"][self.config["DATASET_NAME"]]["TRAIN_START_TIMESTAMP"])
        train_end = pd.to_datetime(self.config["DATASET_SPLIT_CONFIG"][self.config["DATASET_NAME"]]["TRAIN_END_TIMESTAMP"])
        val_start = pd.to_datetime(self.config["DATASET_SPLIT_CONFIG"][self.config["DATASET_NAME"]]["VAL_START_TIMESTAMP"])
        val_end = pd.to_datetime(self.config["DATASET_SPLIT_CONFIG"][self.config["DATASET_NAME"]]["VAL_END_TIMESTAMP"])
        test_start = pd.to_datetime(self.config["DATASET_SPLIT_CONFIG"][self.config["DATASET_NAME"]]["TEST_START_TIMESTAMP"])
        test_end = pd.to_datetime(self.config["DATASET_SPLIT_CONFIG"][self.config["DATASET_NAME"]]["TEST_END_TIMESTAMP"])

        train = df[df.index <= train_end]
        val = df[(df.index >= val_start) & (df.index <= val_end)]
        test = df[(df.index >= test_start) & (df.index <= test_end)]
        return train, val, test

    def label_making(self, raw_df, df):
        raw_df.set_index("DATE", inplace=True)
        df_label = pd.DataFrame({"LABEL": raw_df.CLOSE.shift(-1) / raw_df.CLOSE})
        df_label.iloc[-1, -1] = 1

        df_label = pd.merge(df, df_label, left_index=True, right_index=True, how="outer")
        df_label = df_label.dropna(how="any")
        df_label = df_label["LABEL"].to_frame()
        return df_label

    def load_dataset(self):
        """
        Load the raw dataset and perform some data processing operations
    
        Returns:
        - df (DataFrame): Processed DataFrame
        """
        # print(self.config["MANUAL_SEED"])
        logdir = make_logdir()

        if self.config["LOAD_LOCAL_DATALOADER"]:
            try:
                load_dataset_output = torch.load(ROOT_PATH + "/data/datasets/" + self.config["DATASET_NAME"] + "_load_dataset_output.pt")
                load_dataset_output["logdir"] = logdir
                print("Local dataloader loaded successfully!")
            except Exception:
                self.config["LOAD_LOCAL_DATALOADER"] = False
                update_config(self.config)
                print("Local dataloader does not exist, the config['LOAD_LOCAL_DATALOADER'] is modified as `False` "
                      "automatically")

        # Ensure that the config is updated in all cases
        update_config(self.config)

        # Clear data structures if LOAD_LOCAL_DATALOADER is False
        if not self.config["LOAD_LOCAL_DATALOADER"]:
            ds_train = []
            ds_val = []
            ds_test = []
            label_train = []
            label_val = []
            label_test = []
            df_label_MATLAB = pd.DataFrame()

            raw_files = self.data_accessing(ROOT_PATH + "/data/datasets/" + self.config["DATASET_NAME"])
            if len(raw_files) > 0:
                # feature_engineering(excel_files)
                for i, df in tqdm(enumerate(raw_files), total=len(raw_files), desc="Data Processing"):
                    raw_df = df.copy()
                    df = self.data_cleaning(df)

                    df, DETAILED_FEATURE_LIST, DETAILED_NUM_FEATURES = self.feature_engineering(df)
                    df = self.data_cleaning(df)

                    df, WINDOW_SIZE = self.data_augmentation(df)
                    df = self.data_cleaning(df)

                    df_label = self.label_making(raw_df, df)

                    # df_label_temp = df_label.copy()
                    # test_start = pd.to_datetime(DATASET_SPLIT_CONFIG.get(dataset_name)["TEST_START_TIMESTAMP"])
                    # test_end = pd.to_datetime(DATASET_SPLIT_CONFIG.get(dataset_name)["TEST_END_TIMESTAMP"])
                    # df_label_temp = df_label_temp[(df_label_temp.index >= test_start) & (df_label_temp.index <= test_end)]
                    # df_label_MATLAB = pd.concat([df_label_MATLAB, df_label_temp], axis=1)

                    train, val, test = self.data_splitting(df)
                    train_label, val_label, test_label = self.data_splitting(df_label)

                    zscore_train = self.zscore_calculation(train)
                    train_normalization = self.data_normalization(train, zscore_train)
                    val_normalization = self.data_normalization(val, zscore_train)
                    test_normalization = self.data_normalization(test, zscore_train)

                    nan_count = df.isna().sum().sum()
                    if nan_count != 0:
                        print(f"nan_count: {nan_count}")
                        time.sleep(1111)

                    nan_count = df.isnull().sum().sum()
                    if nan_count != 0:
                        print(f"nan_count: {nan_count}")
                        time.sleep(1111)

                    train_normalization = self.data_cleaning(train_normalization)
                    val_normalization = self.data_cleaning(val_normalization)
                    test_normalization = self.data_cleaning(test_normalization)

                    ds_train.append(torch.from_numpy(train_normalization.values))
                    ds_val.append(torch.from_numpy(val_normalization.values))
                    ds_test.append(torch.from_numpy(test_normalization.values))
                    label_train.append(torch.from_numpy(train_label["LABEL"].values))
                    label_val.append(torch.from_numpy(val_label["LABEL"].values))
                    label_test.append(torch.from_numpy(test_label["LABEL"].values))

                ds_train = torch.stack(ds_train).permute(1, 0, 2).to(self.config["DEVICE"])  # [num_assets, num_train_periods, num_feats] -> [num_train_periods, num_assets, num_feats]
                ds_val = torch.stack(ds_val).permute(1, 0, 2).to(self.config["DEVICE"])
                ds_test = torch.stack(ds_test).permute(1, 0, 2).to(self.config["DEVICE"])
                label_train = torch.stack(label_train).transpose(0, 1).to(self.config["DEVICE"])  # [num_assets, num_train_periods] -> [num_train_periods, num_assets]
                label_val = torch.stack(label_val).transpose(0, 1).to(self.config["DEVICE"])
                label_test = torch.stack(label_test).transpose(0, 1).to(self.config["DEVICE"])

                train_ids = TensorDataset(ds_train, label_train)
                val_ids = TensorDataset(ds_val, label_val)
                test_ids = TensorDataset(ds_test, label_test)

                # Save the price relative tensor to a mat file with the data set name
                # from scipy import io
                # io.savemat("price_relative_" + dataset_name + ".mat", {"data": df_label_MATLAB.values})
            else:
                print("No Excel files found.")

            load_dataset_output = {
                "logdir": logdir,
                "train_loader": DataLoader(train_ids, batch_size=self.config["BATCH_SIZE"][self.config["DATASET_NAME"]], shuffle=False, drop_last=False),
                "val_loader": DataLoader(val_ids, batch_size=self.config["BATCH_SIZE"][self.config["DATASET_NAME"]], shuffle=False, drop_last=False),
                "test_loader": DataLoader(test_ids, batch_size=1, shuffle=False, drop_last=False),
                "test_loader_for_train": DataLoader(test_ids, batch_size=self.config["BATCH_SIZE"][self.config["DATASET_NAME"]], shuffle=False, drop_last=False),
                "NUM_TRAIN_PERIODS": ds_train.shape[0],
                "NUM_VAL_PERIODS": ds_val.shape[0],
                "NUM_TEST_PERIODS": ds_test.shape[0],
                "NUM_ASSETS": ds_train.shape[1],
                "NUM_FEATURES_AUGMENTED": ds_train.shape[2],
                "NUM_FEATURES_ORIGINAL": int(ds_train.shape[2] / (WINDOW_SIZE)),
                "DETAILED_NUM_FEATURES": DETAILED_NUM_FEATURES,
                "WINDOW_SIZE": WINDOW_SIZE,
                "OVERALL_FEATURE_LIST": [feature[8:] for feature, include in self.config["FEATURE_ENGINEERING_CONFIG"].items() if include],
                "DETAILED_FEATURE_LIST": DETAILED_FEATURE_LIST,

            }
            torch.save(load_dataset_output, ROOT_PATH + "/data/datasets/" + self.config["DATASET_NAME"] + "_load_dataset_output.pt")
        print(load_dataset_output)
        return load_dataset_output


if __name__ == "__main__":
    load_dataset_output = DatasetLoader().load_dataset()
    # print(load_dataset_output)