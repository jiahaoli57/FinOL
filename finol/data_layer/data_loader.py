import os
import time
import torch

import json5
import argparse
import pandas as pd
import talib as ta
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from finol import utils
from finol.data_layer.scaler_selector import *
from finol.config import *


def data_accessing(folder_path):
    excel_files = []
    for file_name in tqdm(os.listdir(folder_path), desc="Data Loading"):
        if file_name.endswith('.xlsx') or file_name.endswith('.xls'):
            file_path = os.path.join(folder_path, file_name)
            try:
                dataframe = pd.read_excel(file_path)
                excel_files.append(dataframe)
            except Exception as e:
                print(f"An error occurred while loading file {file_path}: {str(e)}")
        # break
    return excel_files


def feature_engineering(df):
    overlap_features_df = pd.DataFrame()
    momentum_features_df = pd.DataFrame()
    volume_features_df = pd.DataFrame()
    cycle_features_df = pd.DataFrame()
    price_features_df = pd.DataFrame()
    volatility_features_df = pd.DataFrame()
    pattern_features_df = pd.DataFrame()

    if FEATURE_ENGINEERING_CONFIG.get('INCLUDE_OVERLAP_FEATURES', True):
        overlap_features = {
            'BBANDS_UPPER': ta.BBANDS(df.CLOSE)[0],  # Bollinger Bands - Upper Band
            'BBANDS_MIDDLE': ta.BBANDS(df.CLOSE)[1],  # Bollinger Bands - Middle Band
            'BBANDS_LOWER': ta.BBANDS(df.CLOSE)[2],  # Bollinger Bands - Lower Band
            'DEMA': ta.DEMA(df.CLOSE),  # Double Exponential Moving Average
            'EMA': ta.EMA(df.CLOSE),  # Exponential Moving Average
            'HT_TRENDLINE': ta.HT_TRENDLINE(df.CLOSE),  # Hilbert Transform - Instantaneous Trendline
            'KAMA': ta.KAMA(df.CLOSE),  # Kaufman Adaptive Moving Average
            'MA': ta.MA(df.CLOSE),  # Moving Average
            'MAMA': ta.MAMA(df.CLOSE)[0],  # MESA Adaptive Moving Average - MAMA
            'MAMA_FAMA': ta.MAMA(df.CLOSE)[1],  # MESA Adaptive Moving Average - FAMA
            'MAVP': ta.MAVP(df.CLOSE, df.DATE),  # Moving Average with Variable Period
            'MIDPOINT': ta.MIDPOINT(df.CLOSE),  # MidPoint over Period
            'MIDPRICE': ta.MIDPRICE(df.HIGH, df.LOW),  # Midpoint Price over Period
            'SAR': ta.SAR(df.HIGH, df.LOW),  # Parabolic SAR
            'SAREXT': ta.SAREXT(df.HIGH, df.LOW),  # Parabolic SAR - Extended
            'SMA': ta.SMA(df.CLOSE),  # Simple Moving Average
            'T3': ta.T3(df.CLOSE),  # Triple Exponential Moving Average (T3)
            'TEMA': ta.TEMA(df.CLOSE),  # Triple Exponential Moving Average
            'TRIMA': ta.TRIMA(df.CLOSE),  # Triangular Moving Average
            'WMA': ta.WMA(df.CLOSE)  # Weighted Moving Average
        }
        overlap_features_df = pd.DataFrame(overlap_features)

    if FEATURE_ENGINEERING_CONFIG.get('INCLUDE_MOMENTUM_FEATURES', True):
        momentum_features = {
            'ADX': ta.ADX(df.HIGH, df.LOW, df.CLOSE),  # Average Directional Movement Index
            'ADXR': ta.ADXR(df.HIGH, df.LOW, df.CLOSE),  # Average Directional Movement Index Rating
            'APO': ta.APO(df.CLOSE),  # Absolute Price Oscillator
            'AROON_UP': ta.AROON(df.HIGH, df.LOW)[0],  # Aroon Up
            'AROON_DOWN': ta.AROON(df.HIGH, df.LOW)[1],  # Aroon Down
            'AROONOSC': ta.AROONOSC(df.HIGH, df.LOW),  # Aroon Oscillator
            'BOP': ta.BOP(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Balance Of Power
            'CCI': ta.CCI(df.HIGH, df.LOW, df.CLOSE),  # Commodity Channel Index
            'CMO': ta.CMO(df.CLOSE),  # Chande Momentum Oscillator
            'DX': ta.DX(df.HIGH, df.LOW, df.CLOSE),  # Directional Movement Index
            'MACD': ta.MACD(df.CLOSE)[0],  # Moving Average Convergence/Divergence
            'MACD_SIGNAL': ta.MACD(df.CLOSE)[1],  # MACD Signal Line
            'MACD_HIST': ta.MACD(df.CLOSE)[2],  # MACD Histogram
            'MACDEXT': ta.MACDEXT(df.CLOSE)[0],  # MACD with controllable MA type
            'MACDEXT_SIGNAL': ta.MACDEXT(df.CLOSE)[1],  # MACDEXT Signal Line
            'MACDEXT_HIST': ta.MACDEXT(df.CLOSE)[2],  # MACDEXT Histogram
            'MACDFIX': ta.MACDFIX(df.CLOSE)[0],  # Moving Average Convergence/Divergence Fix 12/26
            'MACDFIX_SIGNAL': ta.MACDFIX(df.CLOSE)[1],  # MACDFIX Signal Line
            'MACDFIX_HIST': ta.MACDFIX(df.CLOSE)[2],  # MACDFIX Histogram
            'MFI': ta.MFI(df.HIGH, df.LOW, df.CLOSE, df.VOLUME),  # Money Flow Index
            'MINUS_DI': ta.MINUS_DI(df.HIGH, df.LOW, df.CLOSE),  # Minus Directional Indicator
            'MINUS_DM': ta.MINUS_DM(df.HIGH, df.LOW),  # Minus Directional Movement
            'MOM': ta.MOM(df.CLOSE),  # Momentum
            'PLUS_DI': ta.PLUS_DI(df.HIGH, df.LOW, df.CLOSE),  # Plus Directional Indicator
            'PLUS_DM': ta.PLUS_DM(df.HIGH, df.LOW),  # Plus Directional Movement
            'PPO': ta.PPO(df.CLOSE),  # Percentage Price Oscillator
            'ROC': ta.ROC(df.CLOSE),  # Rate of change: ((price/prevPrice)-1)*100
            'ROCP': ta.ROCP(df.CLOSE),  # Rate of change Percentage: (price-prevPrice)/prevPrice
            'ROCR': ta.ROCR(df.CLOSE),  # Rate of change ratio: (price/prevPrice)
            'ROCR100': ta.ROCR100(df.CLOSE),  # Rate of change ratio 100 scale: (price/prevPrice)*100
            'RSI': ta.RSI(df.CLOSE),  # Relative Strength Index
            'STOCH_K': ta.STOCH(df.HIGH, df.LOW, df.CLOSE)[0],  # Stochastic %K
            'STOCH_D': ta.STOCH(df.HIGH, df.LOW, df.CLOSE)[1],  # Stochastic %D
            'STOCHF_K': ta.STOCHF(df.HIGH, df.LOW, df.CLOSE)[0],  # Stochastic Fast %K
            'STOCHF_D': ta.STOCHF(df.HIGH, df.LOW, df.CLOSE)[1],  # Stochastic Fast %D
            'STOCHRSI_K': ta.STOCHRSI(df.CLOSE)[0],  # Stochastic RSI %K
            'STOCHRSI_D': ta.STOCHRSI(df.CLOSE)[1],  # Stochastic RSI %D
            'TRIX': ta.TRIX(df.CLOSE),  # 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA
            'ULTOSC': ta.ULTOSC(df.HIGH, df.LOW, df.CLOSE),  # Ultimate Oscillator
            'WILLR': ta.WILLR(df.HIGH, df.LOW, df.CLOSE)  # Williams' %R
        }
        momentum_features_df = pd.DataFrame(momentum_features)

    if FEATURE_ENGINEERING_CONFIG.get('INCLUDE_VOLUME_FEATURES', True):
        volume_features = {
            'AD': ta.AD(df.HIGH, df.LOW, df.CLOSE, df.VOLUME),  # Chaikin A/D Line
            'ADOSC': ta.ADOSC(df.HIGH, df.LOW, df.CLOSE, df.VOLUME),  # Chaikin A/D Oscillator
            'OBV': ta.OBV(df.CLOSE, df.VOLUME)  # On Balance Volume
        }
        volume_features_df = pd.DataFrame(volume_features)

    if FEATURE_ENGINEERING_CONFIG.get('INCLUDE_CYCLE_FEATURES', True):
        cycle_features = {
            'HT_DCPERIOD': ta.HT_DCPERIOD(df.CLOSE),  # Hilbert Transform - Dominant Cycle Period
            'HT_DCPHASE': ta.HT_DCPHASE(df.CLOSE),  # Hilbert Transform - Dominant Cycle Phase
            'HT_PHASOR_INPHASE': ta.HT_PHASOR(df.CLOSE)[0],  # Hilbert Transform - Phasor Components
            'HT_PHASOR_QUADRATURE': ta.HT_PHASOR(df.CLOSE)[1],  # Hilbert Transform - Phasor Components
            'HT_SINE_LEADSINE': ta.HT_SINE(df.CLOSE)[0],  # Hilbert Transform - SineWave
            'HT_SINE_SINEWAVE': ta.HT_SINE(df.CLOSE)[1],  # Hilbert Transform - SineWave
            'HT_TRENDMODE': ta.HT_TRENDMODE(df.CLOSE)  # Hilbert Transform - Trend vs Cycle Mode
        }
        cycle_features_df = pd.DataFrame(cycle_features)

    if FEATURE_ENGINEERING_CONFIG.get('INCLUDE_PRICE_FEATURES', True):
        price_features = {
            'AVGPRICE': ta.AVGPRICE(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Average Price
            'MEDPRICE': ta.MEDPRICE(df.HIGH, df.LOW),  # Median Price
            'TYPPRICE': ta.TYPPRICE(df.HIGH, df.LOW, df.CLOSE),  # Typical Price
            'WCLPRICE': ta.WCLPRICE(df.HIGH, df.LOW, df.CLOSE)  # Weighted Close Price
        }
        price_features_df = pd.DataFrame(price_features)

    if FEATURE_ENGINEERING_CONFIG.get('INCLUDE_VOLATILITY_FEATURES', True):
        volatility_features = {
            'ATR': ta.ATR(df.HIGH, df.LOW, df.CLOSE),  # Average True Range
            'NATR': ta.NATR(df.HIGH, df.LOW, df.CLOSE),  # Normalized Average True Range
            'TRANGE': ta.TRANGE(df.HIGH, df.LOW, df.CLOSE)  # True Range
        }
        volatility_features_df = pd.DataFrame(volatility_features)

    if FEATURE_ENGINEERING_CONFIG.get('INCLUDE_PATTERN_FEATURES', True):
        pattern_features = {
            'CDL2CROWS': ta.CDL2CROWS(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Two Crows
            'CDL3BLACKCROWS': ta.CDL3BLACKCROWS(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Three Black Crows
            'CDL3INSIDE': ta.CDL3INSIDE(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Three Inside Up/Down
            'CDL3LINESTRIKE': ta.CDL3LINESTRIKE(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Three-Line Strike
            'CDL3OUTSIDE': ta.CDL3OUTSIDE(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Three Outside Up/Down
            'CDL3STARSINSOUTH': ta.CDL3STARSINSOUTH(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Three Stars In The South
            'CDL3WHITESOLDIERS': ta.CDL3WHITESOLDIERS(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Three Advancing White Soldiers
            'CDLABANDONEDBABY': ta.CDLABANDONEDBABY(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Abandoned Baby
            'CDLADVANCEBLOCK': ta.CDLADVANCEBLOCK(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Advance Block
            'CDLBELTHOLD': ta.CDLBELTHOLD(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Belt-Hold
            'CDLBREAKAWAY': ta.CDLBREAKAWAY(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Breakaway
            'CDLCLOSINGMARUBOZU': ta.CDLCLOSINGMARUBOZU(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Closing Marubozu
            'CDLCONCEALBABYSWALL': ta.CDLCONCEALBABYSWALL(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Concealing Baby Swallow
            'CDLCOUNTERATTACK': ta.CDLCOUNTERATTACK(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Counterattack
            'CDLDARKCLOUDCOVER': ta.CDLDARKCLOUDCOVER(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Dark Cloud Cover
            'CDLDOJI': ta.CDLDOJI(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Doji,
            'CDLDOJISTAR': ta.CDLDOJISTAR(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Doji Star
            'CDLDRAGONFLYDOJI': ta.CDLDRAGONFLYDOJI(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Dragonfly Doji
            'CDLENGULFING': ta.CDLENGULFING(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Engulfing Pattern
            'CDLEVENINGDOJISTAR': ta.CDLEVENINGDOJISTAR(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Evening Doji Star
            'CDLEVENINGSTAR': ta.CDLEVENINGSTAR(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Evening Star
            'CDLGAPSIDESIDEWHITE': ta.CDLGAPSIDESIDEWHITE(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Up/Down-Gap Side-By-Side White Lines
            'CDLGRAVESTONEDOJI': ta.CDLGRAVESTONEDOJI(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Gravestone Doji
            'CDLHAMMER': ta.CDLHAMMER(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Hammer
            'CDLHANGINGMAN': ta.CDLHANGINGMAN(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Hanging Man
            'CDLHARAMI': ta.CDLHARAMI(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Harami Pattern
            'CDLHARAMICROSS': ta.CDLHARAMICROSS(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Harami Cross Pattern
            'CDLHIGHWAVE': ta.CDLHIGHWAVE(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # High-Wave Candle
            'CDLHIKKAKE': ta.CDLHIKKAKE(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Hikkake Pattern
            'CDLHIKKAKEMOD': ta.CDLHIKKAKEMOD(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Modified Hikkake Pattern
            'CDLHOMINGPIGEON': ta.CDLHOMINGPIGEON(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Homing Pigeon
            'CDLIDENTICAL3CROWS': ta.CDLIDENTICAL3CROWS(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Identical Three Crows
            'CDLINNECK': ta.CDLINNECK(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # In-Neck Pattern
            'CDLINVERTEDHAMMER': ta.CDLINVERTEDHAMMER(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Inverted Hammer
            'CDLKICKING': ta.CDLKICKING(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Kicking
            'CDLKICKINGBYLENGTH': ta.CDLKICKINGBYLENGTH(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Kicking - Bull/Bear Determined by the Longer Marubozu
            'CDLLADDERBOTTOM': ta.CDLLADDERBOTTOM(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Ladder Bottom
            'CDLLONGLEGGEDDOJI': ta.CDLLONGLEGGEDDOJI(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Long Legged Doji
            'CDLLONGLINE': ta.CDLLONGLINE(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Long Line Candle
            'CDLMARUBOZU': ta.CDLMARUBOZU(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Marubozu
            'CDLMATCHINGLOW': ta.CDLMATCHINGLOW(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Matching Low
            'CDLMATHOLD': ta.CDLMATHOLD(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Mat Hold
            'CDLMORNINGDOJISTAR': ta.CDLMORNINGDOJISTAR(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Morning Doji Star
            'CDLMORNINGSTAR': ta.CDLMORNINGSTAR(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Morning Star
            'CDLONNECK': ta.CDLONNECK(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # On-Neck Pattern
            'CDLPIERCING': ta.CDLPIERCING(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Piercing Pattern
            'CDLRICKSHAWMAN': ta.CDLRICKSHAWMAN(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Rickshaw Man
            'CDLRISEFALL3METHODS': ta.CDLRISEFALL3METHODS(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Rising/Falling Three Methods
            'CDLSEPARATINGLINES': ta.CDLSEPARATINGLINES(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Separating Lines
            'CDLSHOOTINGSTAR': ta.CDLSHOOTINGSTAR(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Shooting Star
            'CDLSHORTLINE': ta.CDLSHORTLINE(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Short Line Candle
            'CDLSPINNINGTOP': ta.CDLSPINNINGTOP(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Spinning Top
            'CDLSTALLEDPATTERN': ta.CDLSTALLEDPATTERN(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Stalled Pattern
            'CDLSTICKSANDWICH': ta.CDLSTICKSANDWICH(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Stick Sandwich
            'CDLTAKURI': ta.CDLTAKURI(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Takuri (Dragonfly Doji with Very Long Lower Shadow)
            'CDLTASUKIGAP': ta.CDLTASUKIGAP(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Tasuki Gap
            'CDLTHRUSTING': ta.CDLTHRUSTING(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Thrusting Pattern
            'CDLTRISTAR': ta.CDLTRISTAR(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Tristar Pattern
            'CDLUNIQUE3RIVER': ta.CDLUNIQUE3RIVER(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Unique 3 River
            'CDLUPSIDEGAP2CROWS': ta.CDLUPSIDEGAP2CROWS(df.OPEN, df.HIGH, df.LOW, df.CLOSE),  # Upside Gap Two Crows
            'CDLXSIDEGAP3METHODS': ta.CDLXSIDEGAP3METHODS(df.OPEN, df.HIGH, df.LOW, df.CLOSE)  # Upside/Downside Gap Three Methods
        }
        pattern_features_df = pd.DataFrame(pattern_features)

    _ = pd.concat([df, overlap_features_df, momentum_features_df, volume_features_df, cycle_features_df,
                    price_features_df, volatility_features_df, pattern_features_df], axis=1)
    _.set_index('DATE', inplace=True)
    return _


def data_augmentation(df):
    if DATA_AUGMENTATION_CONFIG.get("WINDOW_DATA")["INCLUDE_WINDOW_DATA"]:
        WINDOW_SIZE = DATA_AUGMENTATION_CONFIG.get("WINDOW_DATA")["WINDOW_SIZE"]
        df = pd.concat([df] + [df.shift(i).add_prefix(f'prev_{i}_') for i in range(1, WINDOW_SIZE)], axis=1)
    else:
        WINDOW_SIZE = 1

    return df, WINDOW_SIZE


def data_cleaning(df):
    """
    Clean the DataFrame by removing rows with missing values.
    """
    return df.dropna(how='any')
    # return


def zscore_calculation(df):
    df_normalized = df.copy()
    numeric_features = df.select_dtypes(include=['int', 'float']).columns
    scaler = select_scaler()
    zscore = scaler.fit(df_normalized[numeric_features])  # zscore is different for different assets

    return zscore


def data_normalization(df, zscore):
    """
    Normalize all numeric features in DataFrame
    """
    df_normalized = df.copy()
    scaler = select_scaler()
    numeric_features = df.select_dtypes(include=['int', 'float']).columns
    df_normalized[numeric_features] = zscore.transform(df_normalized[numeric_features])

    return df_normalized


def data_splitting(df):
    """
    Split the DataFrame into train, validation, and test sets.
    """

    # from sklearn.model_selection import train_test_split
    # train, test = train_test_split(df, test_size=TEST_SIZE, shuffle=False)
    # train, val = train_test_split(train, test_size=VAL_SIZE, shuffle=False)
    # print(train)
    # print(val)
    # print(test)
    # time.sleep(1111)

    train_start = pd.to_datetime(DATASET_SPLIT_CONFIG.get(DATASET_NAME)["TRAIN_START_TIMESTAMP"])
    train_end = pd.to_datetime(DATASET_SPLIT_CONFIG.get(DATASET_NAME)["TRAIN_END_TIMESTAMP"])
    val_start = pd.to_datetime(DATASET_SPLIT_CONFIG.get(DATASET_NAME)["VAL_START_TIMESTAMP"])
    val_end = pd.to_datetime(DATASET_SPLIT_CONFIG.get(DATASET_NAME)["VAL_END_TIMESTAMP"])
    test_start = pd.to_datetime(DATASET_SPLIT_CONFIG.get(DATASET_NAME)["TEST_START_TIMESTAMP"])
    test_end = pd.to_datetime(DATASET_SPLIT_CONFIG.get(DATASET_NAME)["TEST_END_TIMESTAMP"])

    train = df[df.index <= train_end]
    val = df[(df.index >= val_start) & (df.index <= val_end)]
    test = df[(df.index >= test_start) & (df.index <= test_end)]
    # print(train)
    # print(val)
    # print(test)
    # time.sleep(1111)
    return train, val, test


def load_dataset():
    """
    Load the raw dataset and perform some data processing operations

    Returns:
    - df (DataFrame): Processed DataFrame
    """

    ds_train = []
    ds_val = []
    ds_test = []
    label_train = []
    label_val = []
    label_test = []
    df_label_MATLAB = pd.DataFrame()

    raw_files = data_accessing(ROOT_PATH + '/data_layer/data/' + DATASET_NAME)
    # print(raw_files[0])
    # df = raw_files[0]
    #
    # # Create figure and axis objects
    # fig, ax1 = plt.subplots()
    #
    # # Convert DATE column to numpy array
    # DATE_values = np.array(df['DATE'].astype(str))
    #
    # # Convert other columns to numpy arrays
    # OPEN_values = np.array(df['OPEN'])
    # HIGH_values = np.array(df['HIGH'])
    # LOW_values = np.array(df['LOW'])
    # CLOSE_values = np.array(df['CLOSE'])
    # VOLUME_values = np.array(df['VOLUME'])
    #
    # # Plot 'OPEN', 'HIGH', 'LOW', 'CLOSE' on the first axis
    # ax1.plot(DATE_values, OPEN_values, color='blue', label='OPEN')
    # ax1.plot(DATE_values, HIGH_values, color='green', label='HIGH')
    # ax1.plot(DATE_values, LOW_values, color='red', label='LOW')
    # ax1.plot(DATE_values, CLOSE_values, color='purple', label='CLOSE')
    #
    # # Set x-axis tick marks and rotate labels
    # plt.xticks(np.arange(0, len(DATE_values), 100), rotation=45)
    #
    # # Set left y-axis label and title
    # ax1.set_ylabel('Price')
    # ax1.set_title("AA")
    #
    # # Create a second axis for VOLUME
    # ax2 = ax1.twinx()
    #
    # # Plot 'VOLUME' on the second axis
    # ax2.plot(DATE_values, VOLUME_values, color='orange', label='VOLUME')
    #
    # # Set right y-axis label
    # ax2.set_ylabel('Volume')
    #
    # # Add legend
    # lines = ax1.get_lines() + ax2.get_lines()
    # labels = [line.get_label() for line in lines]
    # ax1.legend(lines, labels, loc='best')
    #
    # # Adjust the spacing between subplots
    # plt.tight_layout()
    # # Autoformat the date labels
    # plt.xticks(np.arange(0, len(DATE_values), 500), rotation=45)
    # plt.savefig('.eps',
    #             format='eps',
    #             dpi=300,
    #             bbox_inches='tight')
    # # Show the plot
    # plt.show()
    #
    # time.sleep(1111)
    if len(raw_files) > 0:
        print(f"Successfully loaded {len(raw_files)} Excel file(s):")
        # feature_engineering(excel_files)
        for i, df in tqdm(enumerate(raw_files), total=len(raw_files), desc="Data Processing"):
            # print(f"Excel file {i+1}:")
            df = data_cleaning(df)

            df = feature_engineering(df)
            df = data_cleaning(df)

            df, WINDOW_SIZE = data_augmentation(df)
            df = data_cleaning(df)

            df_label = pd.DataFrame({'LABEL': df['CLOSE'].shift(-1) / df['CLOSE']})
            df_label.iloc[-1, -1] = 1

            # df_label_temp = df_label.copy()
            # test_start = pd.to_datetime(DATASET_SPLIT_CONFIG.get(DATASET_NAME)["TEST_START_TIMESTAMP"])
            # test_end = pd.to_datetime(DATASET_SPLIT_CONFIG.get(DATASET_NAME)["TEST_END_TIMESTAMP"])
            # df_label_temp = df_label_temp[(df_label_temp.index >= test_start) & (df_label_temp.index <= test_end)]
            # df_label_MATLAB = pd.concat([df_label_MATLAB, df_label_temp], axis=1)

            train, val, test = data_splitting(df)
            train_label, val_label, test_label = data_splitting(df_label)

            zscore_train = zscore_calculation(train)
            train_normalization = data_normalization(train, zscore_train)
            val_normalization = data_normalization(val, zscore_train)
            test_normalization = data_normalization(test, zscore_train)

            nan_count = df.isna().sum().sum()
            if nan_count != 0:
                print(f'nan_count: {nan_count}')
                time.sleep(1111)

            nan_count = df.isnull().sum().sum()
            if nan_count != 0:
                print(f'nan_count: {nan_count}')
                time.sleep(1111)

            train_normalization = data_cleaning(train_normalization)
            val_normalization = data_cleaning(val_normalization)
            test_normalization = data_cleaning(test_normalization)

            ds_train.append(torch.from_numpy(train_normalization.values))
            ds_val.append(torch.from_numpy(val_normalization.values))
            ds_test.append(torch.from_numpy(test_normalization.values))
            label_train.append(torch.from_numpy(train_label['LABEL'].values))
            label_val.append(torch.from_numpy(val_label['LABEL'].values))
            label_test.append(torch.from_numpy(test_label['LABEL'].values))

        ds_train = torch.stack(ds_train).permute(1, 0, 2)  # [num_assets, num_train_periods, num_feats] -> [num_train_periods, num_assets, num_feats]
        ds_val = torch.stack(ds_val).permute(1, 0, 2)
        ds_test = torch.stack(ds_test).permute(1, 0, 2)
        label_train = torch.stack(label_train).transpose(0, 1)  # [num_assets, num_train_periods] -> [num_train_periods, num_assets]
        label_val = torch.stack(label_val).transpose(0, 1)
        label_test = torch.stack(label_test).transpose(0, 1)

        train_ids = TensorDataset(ds_train, label_train)
        val_ids = TensorDataset(ds_val, label_val)
        test_ids = TensorDataset(ds_test, label_test)

        train_loader = DataLoader(train_ids, batch_size=BATCH_SIZE[DATASET_NAME], shuffle=False, drop_last=False)
        val_loader = DataLoader(val_ids, batch_size=BATCH_SIZE[DATASET_NAME], shuffle=False, drop_last=False)
        test_loader = DataLoader(test_ids, batch_size=1, shuffle=False, drop_last=False)

    else:
        print("No Excel files found.")

    # from scipy import io
    # Save the price relative tensor to a mat file with the data set name
    # io.savemat('price_relative_' + DATASET_NAME + '.mat', {'data': df_label_MATLAB.values})

    load_dataset_output = {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        'NUM_TRAIN_PERIODS': ds_train.shape[0],
        'NUM_VAL_PERIODS': ds_val.shape[0],
        'NUM_TEST_PERIODS': ds_test.shape[0],
        'NUM_ASSETS': ds_train.shape[1],
        'NUM_FEATURES_AUGMENTED': ds_train.shape[2],
        'NUM_FEATURES_ORIGINAL': int(ds_train.shape[2] / (WINDOW_SIZE)),
        'WINDOW_SIZE': WINDOW_SIZE
    }

    return load_dataset_output


if __name__ == '__main__':
    # utils.check_update(GET_LATEST_FINOL)
    load_dataset_output = load_dataset()
    print(load_dataset_output)