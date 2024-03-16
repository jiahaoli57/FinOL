import time

import yfinance as yf
import pandas as pd
from datetime import datetime

pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.max_rows', None)  # 显示所有行
pd.set_option('display.width', None)  # 自动调整列宽以适应数据1
pd.set_option('display.expand_frame_repr', False)  # 不自动换行

DATASET = "CRYPTO"
# ticker_list = ['AA', 'BA', 'BP', 'CAT', 'CNP', 'CVX', 'DIS', 'DTE', 'ED', 'FL', 'GD', 'GE', 'HPQ', 'IBM', 'IP', 'JNJ',
#                  'KO', 'KR', 'MMM', 'MO', 'MRK', 'MRO', 'MSI', 'PG', 'RTX', 'XOM']
# start = '1962-6-10'  # 起点可以前很多
# end = '1985-1-2'  # 终点要多一天
# target_start = '1962-7-3'  # 这天将会是excel的开始日，所以这天要和olps表一样
# #
# ticker_list = ['AA', 'ABM', 'ABT', 'ADM', 'AEM', 'AFG', 'AFL', 'AIG', 'AIR', 'AIT', 'AJG', 'ALE', 'ALK', 'ALX', 'AME', 'AON', 'AOS', 'AP', 'APD', 'ARL', 'ARW', 'ASB', 'ASH', 'ATO', 'AVA', 'AVY', 'AWR', 'AXP', 'AXR', 'AZZ', 'B', 'BA', 'BAC', 'BALL', 'BAX', 'BBWI', 'BC', 'BCE', 'BDX', 'BEN', 'BH', 'BHP', 'BIO', 'BK', 'BKH', 'BMI', 'BMY', 'BN', 'BOH', 'BP', 'BRO', 'BRT', 'BTI', 'BXMT', 'C', 'CACI', 'CAG', 'CAH', 'CAL', 'CAT', 'CBT', 'CCK', 'CDE', 'CFR', 'CHD', 'CHE', 'CI', 'CIA', 'CL', 'CLF', 'CLX', 'CMA', 'CMC', 'CMI', 'CMS', 'CNA', 'CNP', 'COP', 'CP', 'CPB', 'CPK', 'CRS', 'CSL', 'CTO', 'CTS', 'CULP', 'CUZ', 'CVS', 'CVX', 'CW', 'CWT', 'CXT', 'D', 'DCI', 'DCO', 'DD', 'DDS', 'DE', 'DHR', 'DINO', 'DIS', 'DLX', 'DOV', 'DTE', 'DUK', 'DXC', 'DY', 'EAT', 'EBF', 'ECL', 'ED', 'EFX', 'EGP', 'EIX', 'ELME', 'EMR', 'ENB', 'ENZ', 'EQT', 'ES', 'ETN', 'ETR', 'EXPD', 'F', 'FDX', 'FHN', 'FL', 'FLO', 'FLS', 'FMC', 'FRT', 'FSS', 'FUL', 'GATX', 'GBCI', 'GCO', 'GD', 'GE', 'GFF', 'GFI', 'GGG', 'GHC', 'GHM', 'GIS', 'GL', 'GLT', 'GLW', 'GPC', 'GPS', 'GRC', 'GSK', 'GTY', 'GWW', 'HAL', 'HD', 'HE', 'HEI', 'HES', 'HL', 'HMC', 'HNI', 'HOV', 'HP', 'HPQ', 'HRB', 'HRL', 'HSY', 'HUBB', 'HUM', 'HVT', 'HXL', 'IBM', 'IDA', 'IFF', 'IP', 'IPG', 'ITW', 'J', 'JEF', 'JNJ', 'JPM', 'JWN', 'K', 'KAMN', 'KEX', 'KGC', 'KMB', 'KMT', 'KO', 'KR', 'KWR', 'L', 'LEG', 'LEN', 'LHX', 'LLY', 'LMT', 'LNC', 'LOW', 'LPX', 'LUMN', 'LUV', 'LXU', 'LZB', 'MAS', 'MATX', 'MCD', 'MCS', 'MDC', 'MDT', 'MDU', 'MEI', 'MGA', 'MKC', 'MMC', 'MMM', 'MO', 'MOD', 'MRK', 'MRO', 'MSA', 'MSB', 'MSI', 'MTB', 'MTR', 'MTRN', 'MTZ', 'MUR', 'MUX', 'MYE', 'NBR', 'NC', 'NEE', 'NEM', 'NEU', 'NFG', 'NI', 'NJR', 'NKE', 'NL', 'NNN', 'NOC', 'NPK', 'NRT', 'NSC', 'NUE', 'NVO', 'NVRI', 'NWN', 'NX', 'NYT', 'ODC', 'OGE', 'OII', 'OKE', 'OLN', 'OLP', 'OMC', 'OMI', 'OPY', 'ORI', 'OXM', 'OXY', 'PAR', 'PBI', 'PBT', 'PCG', 'PEG', 'PFE', 'PG', 'PGR', 'PH', 'PHG', 'PHI', 'PHM', 'PKE', 'PNC', 'PNM', 'PNR', 'PNW', 'PPG', 'PPL', 'PRG', 'PSA', 'PVH', 'R', 'RAMP', 'RES', 'REX', 'RF', 'RGR', 'RHI', 'RJF', 'RLI', 'ROG', 'ROK', 'ROL', 'RPM', 'RRC', 'RRX', 'RTX', 'RVTY', 'SBR', 'SCI', 'SCL', 'SCX', 'SEE', 'SF', 'SHEL', 'SHW', 'SJT', 'SJW', 'SKY', 'SLB', 'SMP', 'SNA', 'SO', 'SON', 'SONY', 'SPB', 'SPGI', 'SPXC', 'SR', 'SSL', 'STC', 'STT', 'SU', 'SUP', 'SWK', 'SWN', 'SWX', 'SXI', 'SXT', 'SYK', 'SYY', 'T', 'TAP', 'TARO', 'TDS', 'TDW', 'TEVA', 'TEX', 'TFC', 'TFX', 'TGNA', 'TGT', 'THC', 'THO', 'TISI', 'TKR', 'TM', 'TMO', 'TNC', 'TPC', 'TPL', 'TR', 'TRC', 'TRN', 'TRP', 'TRV', 'TSN', 'TT', 'TTC', 'TXT', 'TYL'
#  , 'UDR', 'UFI', 'UGI', 'UHS', 'UIS', 'UL', 'UNF', 'UNH', 'UNP', 'USB', 'UVV', 'VFC', 'VHI', 'VLO', 'VMC', 'VMI', 'VNO', 'VSH', 'VZ', 'WEC', 'WELL', 'WFC', 'WGO', 'WHR', 'WLY', 'WLYB', 'WMB', 'WMK', 'WMT', 'WOR', 'WRB', 'WSM', 'WSO', 'WST', 'WTRG', 'WWW', 'WY', 'XOM']

# start = '1984-12-29'  # 起点可以前很多
# end = '2010-7-1'  # 终点要多一天
# target_start = '1985-1-2'  # 这天将会是excel的开始日，所以这天要和olps表一样
#
# ticker_list = ['A', 'AAPL', 'AMGN', 'AXP', 'B', 'BA', 'CAT', 'CSCO', 'CVX', 'DIS', 'GS', 'HD', 'HON', 'IBM', 'INTC',
#                'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT', 'PG', 'TRV', 'UNH', 'VZ', 'WBA', 'WMT']

# start = '2001-1-1'  # 起点可以前很多
# end = '2003-1-15'  # 终点要多一天
# target_start = '2001-1-14'  # 这天将会是excel的开始日，所以这天要和olps表一样
# num_periods = 500
#
# ticker_list = ['AAPL', 'ABT', 'ACGL', 'ADBE', 'ADI', 'ADM', 'ADP', 'ADSK', 'AEP', 'AES', 'AFL', 'AIG', 'AJG', 'ALB', 'ALK', 'ALL', 'AMAT', 'AMD', 'AME', 'AMGN', 'AMZN', 'ANSS', 'AON', 'AOS', 'APA', 'APD', 'APH', 'ARE', 'ATO', 'AVB', 'AVY', 'AXP', 'AZO', 'BA', 'BAC', 'BALL', 'BAX', 'BBWI', 'BBY', 'BDX', 'BEN', 'BIIB', 'BIO', 'BK', 'BKR', 'BMY', 'BRO', 'BSX', 'BWA', 'BXP', 'C', 'CAG', 'CAH', 'CAT', 'CB', 'CCL', 'CDNS', 'CHD', 'CHRW', 'CI', 'CINF', 'CL', 'CLX', 'CMA', 'CMCSA', 'CMI', 'CMS', 'CNP', 'COF', 'COO', 'COP', 'COST', 'CPB', 'CPRT', 'CPT', 'CSCO', 'CSX', 'CTAS', 'CTRA', 'CVS', 'CVX', 'D', 'DD', 'DE', 'DGX', 'DHI', 'DHR', 'DIS', 'DLTR', 'DOV', 'DRI', 'DTE', 'DUK', 'DVA', 'DVN', 'DXC', 'EA', 'ECL', 'ED', 'EFX', 'EG', 'EIX', 'EL', 'EMN', 'EMR', 'EOG', 'EQR', 'EQT', 'ES', 'ESS', 'ETN', 'ETR', 'EVRG', 'EXC', 'EXPD', 'F', 'FAST', 'FCX', 'FDS', 'FDX', 'FE', 'FI', 'FICO', 'FITB', 'FMC', 'FRT', 'GD', 'GE', 'GEN', 'GILD', 'GIS', 'GL', 'GLW', 'GPC', 'GWW', 'HAL', 'HAS', 'HBAN', 'HD', 'HES', 'HIG', 'HOLX', 'HON', 'HPQ', 'HRL', 'HSIC', 'HST', 'HSY', 'HUM', 'IBM', 'IDXX', 'IEX', 'IFF', 'INCY', 'INTC', 'INTU', 'IP', 'IPG', 'IRM', 'IT', 'ITW', 'IVZ', 'J', 'JBHT', 'JCI', 'JKHY', 'JNJ', 'JPM', 'K', 'KEY', 'KIM', 'KLAC', 'KMB', 'KMX', 'KO', 'KR', 'L', 'LEN', 'LH', 'LHX', 'LIN', 'LLY', 'LMT', 'LNC', 'LNT', 'LOW', 'LRCX', 'LUV', 'MAA', 'MAS', 'MCD', 'MCHP', 'MCK', 'MCO', 'MDT', 'MGM', 'MHK', 'MKC', 'MLM', 'MMC', 'MMM', 'MNST', 'MO', 'MOS', 'MRK', 'MRO', 'MS', 'MSFT', 'MSI', 'MTB', 'MTCH', 'MTD', 'MU', 'NDSN', 'NEE', 'NEM', 'NI', 'NKE', 'NOC', 'NSC', 'NTAP', 'NTRS', 'NUE', 'NVR', 'NWL', 'O', 'ODFL', 'OKE', 'OMC', 'ORCL', 'ORLY', 'OXY', 'PAYX', 'PCAR', 'PCG', 'PEAK', 'PEG', 'PEP', 'PFE', 'PG', 'PGR', 'PH', 'PHM', 'PLD', 'PNC', 'PNR', 'PNW', 'POOL', 'PPG', 'PPL', 'PSA', 'PTC', 'PXD', 'QCOM', 'RCL', 'REG', 'REGN', 'RF', 'RHI', 'RJF', 'RL', 'RMD', 'ROK', 'ROL', 'ROP', 'ROST', 'RTX', 'RVTY', 'SBUX', 'SCHW', 'SEE', 'SHW', 'SJM', 'SLB', 'SNA', 'SNPS', 'SO', 'SPG', 'SPGI', 'STE', 'STLD', 'STT', 'STZ', 'SWK', 'SWKS', 'SYK', 'SYY', 'T', 'TAP', 'TECH', 'TER', 'TFC', 'TFX', 'TGT', 'TJX', 'TMO', 'TRMB', 'TROW', 'TRV', 'TSCO', 'TSN', 'TT', 'TTWO', 'TXN', 'TXT', 'TYL', 'UDR', 'UHS', 'UNH', 'UNP', 'URI', 'USB', 'VFC', 'VLO', 'VMC', 'VRTX', 'VTR', 'VTRS', 'VZ', 'WAB', 'WAT', 'WBA', 'WDC', 'WEC', 'WELL', 'WFC', 'WHR', 'WM', 'WMB', 'WMT', 'WRB', 'WST', 'WY', 'XEL', 'XOM', 'XRAY', 'YUM', 'ZBRA', 'ZION']
 # start = '1997-12-28'  # 起点可以前很多
# end = '2003-2-1'  # 终点要多一天
# target_start = '1998-1-2'  # 这天将会是excel的开始日，所以这天要和olps表一样
# num_periods = 1277
# #
# if DATASET == "TSE":
#     ticker_list = ['ABX.TO', 'ACO-X.TO', 'AEM.TO', 'BBD-B.TO', 'BMO.TO', 'BN.TO', 'BNS.TO', 'CAE.TO', 'CCL-B.TO', 'CCO.TO', 'CFP.TO', 'CM.TO', 'CNQ.TO', 'CTC-A.TO', 'DPM.TO', 'EMA.TO', 'EMP-A.TO', 'ENB.TO', 'ERF.TO', 'FFH.TO', 'FTS.TO', 'FTT.TO', 'FVI.TO', 'IFP.TO', 'IGM.TO', 'IMO.TO', 'L.TO', 'LB.TO', 'LNR.TO', 'MATR.TO', 'MRU.TO', 'MX.TO', 'NA.TO', 'ONEX.TO', 'POU.TO', 'POW.TO', 'PRMW.TO', 'QBR-B.TO', 'RCI-B.TO', 'RY.TO', 'SU.TO', 'T.TO', 'TA.TO', 'TD.TO', 'TECK-B.TO', 'TRP.TO', 'WFG.TO', 'WN.TO']
#

#     start = '1995-01-12'  # 起点可以前很多
#     end = '1999-1-3'  # 终点要多一天
#     target_start = '1995-01-12'  # 这天将会是excel的开始日，所以这天要和olps表一样
#     num_periods = 1001
#
# if DATASET == "SSE":
#     ticker_list = ['600010.SS', '600028.SS', '600030.SS', '600031.SS', '600048.SS', '600050.SS', '600089.SS', '600104.SS',
#      '600111.SS', '600196.SS', '600276.SS', '600309.SS', '600406.SS', '600436.SS', '600438.SS', '600519.SS',
#      '600690.SS', '600745.SS', '600809.SS', '600887.SS', '600900.SS', '601088.SS', '601166.SS', '601318.SS',
#      '601390.SS', '601398.SS', '601628.SS', '601857.SS', '601899.SS', '601919.SS']
#     start = '2000-07-01'  # 起点可以前很多
#     end = '2023-06-30'  # 终点要多一天
#     target_start = '2010-07-05'  # 这天将会是excel的开始日，所以这天要和olps表一样
#     num_periods = 678
#
# if DATASET == "HSI":
#     ticker_list = [
#         "0241.HK","2020.HK","3988.HK","2388.HK","1211.HK","0939.HK","2628.HK",
#         "2319.HK","3968.HK","0941.HK","0688.HK","0386.HK","0291.HK","1109.HK",
#         "0836.HK","1088.HK","0762.HK","0267.HK","0001.HK","1038.HK","0002.HK",
#         "0883.HK","2007.HK","1093.HK","2688.HK","0027.HK","0175.HK","0101.HK",
#         "0011.HK","0012.HK","1044.HK","0003.HK","0388.HK","0005.HK","1398.HK",
#         "0992.HK","2331.HK","0823.HK","0066.HK","0017.HK","0316.HK","0857.HK",
#         "0006.HK","2313.HK","0016.HK","1177.HK","0981.HK","2382.HK","0669.HK",
#         "0700.HK","0322.HK","0868.HK","2899.HK"
#     ]
#     start = '2000-07-01'  # 起点可以前很多
#     end = '2023-06-30'  # 终点要多一天
#     target_start = '2010-07-05'  # 这天将会是excel的开始日，所以这天要和olps表一样
#     num_periods = 678

if DATASET == "CMEG":
    ticker_list = [
        'ZC=F', 'ZS=F', 'ZW=F', 'CT=F',
        'CL=F', 'BZ=F', 'NG=F', 'RB=F', 'HO=F',
        'ES=F', 'NQ=F', 'YM=F',
        '6E=F', '6J=F', '6B=F', '6A=F', '6S=F', '6M=F', '6N=F',
        'ZT=F', 'ZF=F', 'ZN=F',
        'GC=F', 'SI=F', 'HG=F'
    ]
    start = '2000-07-01'  # 起点可以前很多
    end = '2023-06-30'  # 终点要多一天
    target_start = '2010-07-05'  # 这天将会是excel的开始日，所以这天要和olps表一样
    num_periods = 678

if DATASET == "CRYPTO":
    ticker_list = ['ADA-USD', 'ANT-USD', 'BAT-USD', 'BCH-USD', 'BNB-USD', 'BTC-USD', 'BTG-USD', 'DASH-USD', 'DCR-USD', 'DGB-USD', 'DOGE-USD', 'ENJ-USD', 'EOS-USD', 'ETC-USD', 'ETH-USD', 'GAS-USD', 'GLM-USD', 'GNO-USD', 'ICX-USD', 'KCS-USD', 'LINK-USD', 'LRC-USD', 'LSK-USD', 'LTC-USD', 'MANA-USD', 'NEO-USD', 'NMR-USD', 'POWR-USD', 'QTUM-USD', 'RLC-USD', 'SC-USD', 'STORJ-USD', 'STRAX-USD', 'TRX-USD', 'USDT-USD', 'WAVES-USD', 'XEM-USD', 'XLM-USD', 'XMR-USD', 'XRP-USD', 'XTZ-USD', 'ZEC-USD', 'ZRX-USD']
    start = '2017-11-09'  # 起点可以前很多
    end = '2024-03-02'  # 终点要多一天
    target_start = '2017-11-09'  # 这天将会是excel的开始日，所以这天要和olps表一样
    num_periods = 2305

    print(len(ticker_list))
    time.sleep(111)




for ticker in ticker_list:
    # df = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False, keepna=True)  # NYSE(O) Period: Jul 3, 1962 to Dec 31, 1984
    if DATASET == "SSE" or DATASET == "HSI" or DATASET == "CMEG" :
        df = yf.Ticker(ticker).history(start=start, end=end, interval="1wk", auto_adjust=False, keepna=True)  # NYSE(O) Period: Jul 3, 1962 to Dec 31, 1984
        # print(df)
    else:
        df = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False, keepna=True)  # NYSE(O) Period: Jul 3, 1962 to Dec 31, 1984

    # print(tickers_hist)

    df = df.reset_index().rename(columns={'index': 'Date'})
    # print(tickers_hist)

    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    df.columns = ['DATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']

    df['DATE'] = df['DATE'].dt.tz_localize(None)
    df['DATE'] = pd.to_datetime(df['DATE']).dt.date
    # tickers_hist['DATE'] = tickers_hist['DATE'].dt.strftime("%Y/%m/%d")
    # print(tickers_hist['DATE'])
    # tickers_hist['DATE'] = tickers_hist.to_datetime(tickers_hist['DATE'])

    df.set_index('DATE', inplace=True)

    if DATASET != "TSE" and DATASET != "CRYPTO":
        df.iat[0, df.columns.get_loc('OPEN')] = 0
        mask = df['OPEN'] == 0
        df.loc[mask, 'OPEN'] = df.loc[mask, 'CLOSE'].shift(1)

        df.iat[0, df.columns.get_loc('CLOSE')] = 0
        mask = df['CLOSE'] == 0
        df.loc[mask, 'CLOSE'] = df.loc[mask, 'OPEN'].shift(-1)
    # print(df1['CLOSE'].equals(df2['CLOSE']))
    # print(df['OPEN'].equals(df1['OPEN']))
    # time.sleep(5)

    # print(df)
    target_date = pd.to_datetime(target_start).date()
    df = df[df.index >= target_date]

    if df.isna().any().any():
        # print(df.shape)
        # print(ticker)
        # df.to_excel(ticker + '.xlsx')
        # raise ValueError("DataFrame中存在NaN值")
        df = df.dropna(how='any')
    else:
        pass
        # print("DataFrame中没有NaN值")
    # 要确保 OHLC != 0

    if df.shape[0] != num_periods:
        print(ticker)
        print(df.shape)
        # raise ValueError("和前人不符")

    if df.isna().any().any():
        raise ValueError("DataFrame中存在NaN值")
    else:
        df.to_excel(ticker + '.xlsx')
        pass

#
#

# 读取包含股票数据的txt文件
with open('crypto_2024.txt', 'r') as file:
    data = file.readlines()
    print(f"INDEX 共计股票数目: {len(data)}")

# 存储符合条件的股票代码
selected_stocks = []

# 定义目标日期时间
target_date = datetime(1962, 7, 1)  # nyseo
target_date = datetime(1985, 1, 1)  # nysen
target_date = datetime(2001, 1, 1)  # djia
target_date = datetime(1998, 1, 1)  # sp500
target_date = datetime(1995, 1, 13)  # tse
target_date = datetime(2017, 11, 10)  # crypto

# 挑选出目标日期时间之前的所有行，并提取股票代码
for line in data:
    stock, datetime_str = line.strip().split(',')
    stock = stock.replace('.N', '')  # 去除.N后缀
    stock = stock.replace('.O', '')  # 去除.N后缀
    dt = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
    if dt < target_date:
        selected_stocks.append(stock)

selected_stocks.sort()
print(len(selected_stocks))
# 打印符合条件的股票代码
print(selected_stocks)