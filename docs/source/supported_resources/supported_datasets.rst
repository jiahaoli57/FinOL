.. _supported_datasets:

FinOL Datasets
========

.. container::

   +--------------------------+----------------+-----------------+-----------+--------+---------------+--------------------+
   | Name                     | Market         | Country/Region  | Data      | # of   | Data Range    | # of               |
   |                          |                |                 | Frequency | Assets |               | Periods            |
   +==========================+================+=================+===========+========+===============+====================+
   | `NYSE(O)                 | Stock          | United States   | Daily     | 26     | 03/July./1962 | 5,651:             |
   | <https://github.com/ai   |                |                 |           |        | -             | 3,390/1,130/1,131  |
   | 4finol/FinOL_data/tree/m |                |                 |           |        | 31/Dec./1984  |                    |
   | ain/datasets/NYSE(O)>`__ |                |                 |           |        |               |                    |
   |                          |                |                 |           |        |               |                    |
   +--------------------------+----------------+-----------------+-----------+--------+---------------+--------------------+
   | `NYSE(N)                 | Stock          | United States   | Daily     | 403    | 02/Jan./1985  | 6,430:             |
   | <https://github.com/ai   |                |                 |           |        | -             | 3,858/1,286/1,286  |
   | 4finol/FinOL_data/tree/m |                |                 |           |        | 30/June./2010 |                    |
   | ain/datasets/NYSE(N)>`__ |                |                 |           |        |               |                    |
   |                          |                |                 |           |        |               |                    |
   +--------------------------+----------------+-----------------+-----------+--------+---------------+--------------------+
   | `DJIA                    | Stock          | United States   | Daily     | 28     | 14/Jan./2001  | 500:               |
   | <https://github.com      |                |                 |           |        | -             | 300/100/100        |
   | /ai4finol/FinOL_data/tre |                |                 |           |        | 14/Jan./2003  |                    |
   | e/main/datasets/DJIA>`__ |                |                 |           |        |               |                    |
   |                          |                |                 |           |        |               |                    |
   +--------------------------+----------------+-----------------+-----------+--------+---------------+--------------------+
   | `SP500                   | Stock          | United States   | Daily     | 339    | 02/Jan./1998  | 1,268:             |
   | <https://github.com/     |                |                 |           |        | -             | 756/256/256        |
   | ai4finol/FinOL_data/tree |                |                 |           |        | 31/Jan./2003  |                    |
   | /main/datasets/SP500>`__ |                |                 |           |        |               |                    |
   |                          |                |                 |           |        |               |                    |
   +--------------------------+----------------+-----------------+-----------+--------+---------------+--------------------+
   | `TSE <https://github.co  | Stock          | Canada          | Daily     | 48     | 12/Jan./1995  | 1,001:             |
   | m/ai4finol/FinOL_data/tr |                |                 |           |        | -             | 600/200/200        |
   | ee/main/datasets/TSE>`__ |                |                 |           |        | 31/Dec./1998  |                    |
   |                          |                |                 |           |        |               |                    |
   +--------------------------+----------------+-----------------+-----------+--------+---------------+--------------------+
   |                          |                |                 |           |        |               |                    |
   +--------------------------+----------------+-----------------+-----------+--------+---------------+--------------------+
   | `SSE <https://github.co  | Stock          | China           | Weekly    | 30     | 05/July./2010 | 678:               |
   | m/ai4finol/FinOL_data/tr |                |                 |           |        | -             | 406/136/136        |
   | ee/main/datasets/SSE>`__ |                |                 |           |        | 26/June./2023 |                    |
   |                          |                |                 |           |        |               |                    |
   +--------------------------+----------------+-----------------+-----------+--------+---------------+--------------------+
   | `HSI <https://github.co  | Stock          | Hong Kong, China| Weekly    | 53     | 05/July./2010 | 678:               |
   | m/ai4finol/FinOL_data/tr |                |                 |           |        | -             | 406/136/136        |
   | ee/main/datasets/HSI>`__ |                |                 |           |        | 26/June./2023 |                    |
   |                          |                |                 |           |        |               |                    |
   +--------------------------+----------------+-----------------+-----------+--------+---------------+--------------------+
   | `CMEG                    | Futures        | United States   | Weekly    | 25     | 05/July./2010 | 678:               |
   | <https://github.com      |                |                 |           |        | -             | 406/136/136        |
   | /ai4finol/FinOL_data/tre |                |                 |           |        | 26/June./2023 |                    |
   | e/main/datasets/CMEG>`__ |                |                 |           |        |               |                    |
   |                          |                |                 |           |        |               |                    |
   +--------------------------+----------------+-----------------+-----------+--------+---------------+--------------------+
   | `CRYPTO                  | Cryptocurrency | World           | Daily     | 43     | 09/Nov./2017  | 2,305:             |
   | <https://github.com/a    |                |                 |           |        | -             | 1,421/442/442      |
   | i4finol/FinOL_data/tree/ |                |                 |           |        | 01/Mar./2024  |                    |
   | main/datasets/CRYPTO>`__ |                |                 |           |        |               |                    |
   |                          |                |                 |           |        |               |                    |
   +--------------------------+----------------+-----------------+-----------+--------+---------------+--------------------+


Components of Datasets
----------------------

This section presents the complete list of components for each dataset included in
``FinOL``. The assets cover major global equity, futures, and other financial markets.

Components of NYSE(O) Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

	NYSE(O) = ['AA', 'BA', 'BP', 'CAT', 'CNP', 'CVX', 'DIS', 'DTE', 'ED', 'FL', 'GD', 'GE', 'HPQ', 'IBM', 'IP', 'JNJ',  'KO', 'KR', 'MMM', 'MO', 'MRK', 'MRO', 'MSI', 'PG', 'RTX', 'XOM']  # 26 assets

Components of NYSE(N) Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

	NYSE(N) = ['AA', 'ABM', 'ABT', 'ADM', 'AEM', 'AFG', 'AFL', 'AIG', 'AIR', 'AIT', 'AJG', 'ALE', 'ALK', 'ALX', 'AME', 'AON', 'AOS', 'AP', 'APD', 'ARL', 'ARW', 'ASB', 'ASH', 'ATO', 'AVA', 'AVY', 'AWR', 'AXP', 'AXR', 'AZZ', 'B', 'BA', 'BAC', 'BALL', 'BAX', 'BBWI', 'BC', 'BCE', 'BDX', 'BEN', 'BH', 'BHP', 'BIO', 'BK', 'BKH', 'BMI', 'BMY', 'BN', 'BOH', 'BP', 'BRO', 'BRT', 'BTI', 'BXMT', 'C', 'CACI', 'CAG', 'CAH', 'CAL', 'CAT', 'CBT', 'CCK', 'CDE', 'CFR', 'CHD', 'CHE', 'CI', 'CIA', 'CL', 'CLF', 'CLX', 'CMA', 'CMC', 'CMI', 'CMS', 'CNA', 'CNP', 'COP', 'CP', 'CPB', 'CPK', 'CRS', 'CSL', 'CTO', 'CTS', 'CULP', 'CUZ', 'CVS', 'CVX', 'CW', 'CWT', 'CXT', 'D', 'DCI', 'DCO', 'DD', 'DDS', 'DE', 'DHR', 'DINO', 'DIS', 'DLX', 'DOV', 'DTE', 'DUK', 'DXC', 'DY', 'EAT', 'EBF', 'ECL', 'ED', 'EFX', 'EGP', 'EIX', 'ELME', 'EMR', 'ENB', 'ENZ', 'EQT', 'ES', 'ETN', 'ETR', 'EXPD', 'F', 'FDX', 'FHN', 'FL', 'FLO', 'FLS', 'FMC', 'FRT', 'FSS', 'FUL', 'GATX', 'GBCI', 'GCO', 'GD', 'GE', 'GFF', 'GFI', 'GGG', 'GHC', 'GHM', 'GIS', 'GL', 'GLT', 'GLW', 'GPC', 'GPS', 'GRC', 'GSK', 'GTY', 'GWW', 'HAL', 'HD', 'HE', 'HEI', 'HES', 'HL', 'HMC', 'HNI', 'HOV', 'HP', 'HPQ', 'HRB', 'HRL', 'HSY', 'HUBB', 'HUM', 'HVT', 'HXL', 'IBM', 'IDA', 'IFF', 'IP', 'IPG', 'ITW', 'J', 'JEF', 'JNJ', 'JPM', 'JWN', 'K', 'KAMN', 'KEX', 'KGC', 'KMB', 'KMT', 'KO', 'KR', 'KWR', 'L', 'LEG', 'LEN', 'LHX', 'LLY', 'LMT', 'LNC', 'LOW', 'LPX', 'LUMN', 'LUV', 'LXU', 'LZB', 'MAS', 'MATX', 'MCD', 'MCS', 'MDC', 'MDT', 'MDU', 'MEI', 'MGA', 'MKC', 'MMC', 'MMM', 'MO', 'MOD', 'MRK', 'MRO', 'MSA', 'MSB', 'MSI', 'MTB', 'MTR', 'MTRN', 'MTZ', 'MUR', 'MUX', 'MYE', 'NBR', 'NC', 'NEE', 'NEM', 'NEU', 'NFG', 'NI', 'NJR', 'NKE', 'NL', 'NNN', 'NOC', 'NPK', 'NRT', 'NSC', 'NUE', 'NVO', 'NVRI', 'NWN', 'NX', 'NYT', 'ODC', 'OGE', 'OII', 'OKE', 'OLN', 'OLP', 'OMC', 'OMI', 'OPY', 'ORI', 'OXM', 'OXY', 'PAR', 'PBI', 'PBT', 'PCG', 'PEG', 'PFE', 'PG', 'PGR', 'PH', 'PHG', 'PHI', 'PHM', 'PKE', 'PNC', 'PNM', 'PNR', 'PNW', 'PPG', 'PPL', 'PRG', 'PSA', 'PVH', 'R', 'RAMP', 'RES', 'REX', 'RF', 'RGR', 'RHI', 'RJF', 'RLI', 'ROG', 'ROK', 'ROL', 'RPM', 'RRC', 'RRX', 'RTX', 'RVTY', 'SBR', 'SCI', 'SCL', 'SCX', 'SEE', 'SF', 'SHEL', 'SHW', 'SJT', 'SJW', 'SKY', 'SLB', 'SMP', 'SNA', 'SO', 'SON', 'SONY', 'SPB', 'SPGI', 'SPXC', 'SR', 'SSL', 'STC', 'STT', 'SU', 'SUP', 'SWK', 'SWN', 'SWX', 'SXI', 'SXT', 'SYK', 'SYY', 'T', 'TAP', 'TARO', 'TDS', 'TDW', 'TEVA', 'TEX', 'TFC', 'TFX', 'TGNA', 'TGT', 'THC', 'THO', 'TISI', 'TKR', 'TM', 'TMO', 'TNC', 'TPC', 'TPL', 'TR', 'TRC', 'TRN', 'TRP', 'TRV', 'TSN', 'TT', 'TTC', 'TXT', 'TYL' , 'UDR', 'UFI', 'UGI', 'UHS', 'UIS', 'UL', 'UNF', 'UNH', 'UNP', 'USB', 'UVV', 'VFC', 'VHI', 'VLO', 'VMC', 'VMI', 'VNO', 'VSH', 'VZ', 'WEC', 'WELL', 'WFC', 'WGO', 'WHR', 'WLY', 'WLYB', 'WMB', 'WMK', 'WMT', 'WOR', 'WRB', 'WSM', 'WSO', 'WST', 'WTRG', 'WWW', 'WY', 'XOM']  # 403 assets

Components of DJIA Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

	DJIA = ['A', 'AAPL', 'AMGN', 'AXP', 'B', 'BA', 'CAT', 'CSCO', 'CVX', 'DIS', 'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT', 'PG', 'TRV', 'UNH', 'VZ', 'WBA', 'WMT']  # 28 assets

Components of SP500 Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

	SP500 = ['AAPL', 'ABT', 'ACGL', 'ADBE', 'ADI', 'ADM', 'ADP', 'ADSK', 'AEP', 'AES', 'AFL', 'AIG', 'AJG', 'ALB', 'ALK', 'ALL', 'AMAT', 'AMD', 'AME', 'AMGN', 'AMZN', 'ANSS', 'AON', 'AOS', 'APA', 'APD', 'APH', 'ARE', 'ATO', 'AVB', 'AVY', 'AXP', 'AZO', 'BA', 'BAC', 'BALL', 'BAX', 'BBWI', 'BBY', 'BDX', 'BEN', 'BIIB', 'BIO', 'BK', 'BKR', 'BMY', 'BRO', 'BSX', 'BWA', 'BXP', 'C', 'CAG', 'CAH', 'CAT', 'CB', 'CCL', 'CDNS', 'CHD', 'CHRW', 'CI', 'CINF', 'CL', 'CLX', 'CMA', 'CMCSA', 'CMI', 'CMS', 'CNP', 'COF', 'COO', 'COP', 'COST', 'CPB', 'CPRT', 'CPT', 'CSCO', 'CSX', 'CTAS', 'CTRA', 'CVS', 'CVX', 'D', 'DD', 'DE', 'DGX', 'DHI', 'DHR', 'DIS', 'DLTR', 'DOV', 'DRI', 'DTE', 'DUK', 'DVA', 'DVN', 'DXC', 'EA', 'ECL', 'ED', 'EFX', 'EG', 'EIX', 'EL', 'EMN', 'EMR', 'EOG', 'EQR', 'EQT', 'ES', 'ESS', 'ETN', 'ETR', 'EVRG', 'EXC', 'EXPD', 'F', 'FAST', 'FCX', 'FDS', 'FDX', 'FE', 'FI', 'FICO', 'FITB', 'FMC', 'FRT', 'GD', 'GE', 'GEN', 'GILD', 'GIS', 'GL', 'GLW', 'GPC', 'GWW', 'HAL', 'HAS', 'HBAN', 'HD', 'HES', 'HIG', 'HOLX', 'HON', 'HPQ', 'HRL', 'HSIC', 'HST', 'HSY', 'HUM', 'IBM', 'IDXX', 'IEX', 'IFF', 'INCY', 'INTC', 'INTU', 'IP', 'IPG', 'IRM', 'IT', 'ITW', 'IVZ', 'J', 'JBHT', 'JCI', 'JKHY', 'JNJ', 'JPM', 'K', 'KEY', 'KIM', 'KLAC', 'KMB', 'KMX', 'KO', 'KR', 'L', 'LEN', 'LH', 'LHX', 'LIN', 'LLY', 'LMT', 'LNC', 'LNT', 'LOW', 'LRCX', 'LUV', 'MAA', 'MAS', 'MCD', 'MCHP', 'MCK', 'MCO', 'MDT', 'MGM', 'MHK', 'MKC', 'MLM', 'MMC', 'MMM', 'MNST', 'MO', 'MOS', 'MRK', 'MRO', 'MS', 'MSFT', 'MSI', 'MTB', 'MTCH', 'MTD', 'MU', 'NDSN', 'NEE', 'NEM', 'NI', 'NKE', 'NOC', 'NSC', 'NTAP', 'NTRS', 'NUE', 'NVR', 'NWL', 'O', 'ODFL', 'OKE', 'OMC', 'ORCL', 'ORLY', 'OXY', 'PAYX', 'PCAR', 'PCG', 'PEAK', 'PEG', 'PEP', 'PFE', 'PG', 'PGR', 'PH', 'PHM', 'PLD', 'PNC', 'PNR', 'PNW', 'POOL', 'PPG', 'PPL', 'PSA', 'PTC', 'PXD', 'QCOM', 'RCL', 'REG', 'REGN', 'RF', 'RHI', 'RJF', 'RL', 'RMD', 'ROK', 'ROL', 'ROP', 'ROST', 'RTX', 'RVTY', 'SBUX', 'SCHW', 'SEE', 'SHW', 'SJM', 'SLB', 'SNA', 'SNPS', 'SO', 'SPG', 'SPGI', 'STE', 'STLD', 'STT', 'STZ', 'SWK', 'SWKS', 'SYK', 'SYY', 'T', 'TAP', 'TECH', 'TER', 'TFC', 'TFX', 'TGT', 'TJX', 'TMO', 'TRMB', 'TROW', 'TRV', 'TSCO', 'TSN', 'TT', 'TTWO', 'TXN', 'TXT', 'TYL', 'UDR', 'UHS', 'UNH', 'UNP', 'URI', 'USB', 'VFC', 'VLO', 'VMC', 'VRTX', 'VTR', 'VTRS', 'VZ', 'WAB', 'WAT', 'WBA', 'WDC', 'WEC', 'WELL', 'WFC', 'WHR', 'WM', 'WMB', 'WMT', 'WRB', 'WST', 'WY', 'XEL', 'XOM', 'XRAY', 'YUM', 'ZBRA', 'ZION']  # 339 assets

Components of TSE Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

	TSE = ['ABX.TO', 'ACO-X.TO', 'AEM.TO', 'BBD-B.TO', 'BMO.TO', 'BN.TO', 'BNS.TO', 'CAE.TO', 'CCL-B.TO', 'CCO.TO', 'CFP.TO', 'CM.TO', 'CNQ.TO', 'CTC-A.TO', 'DPM.TO', 'EMA.TO', 'EMP-A.TO', 'ENB.TO', 'ERF.TO', 'FFH.TO', 'FTS.TO', 'FTT.TO', 'FVI.TO', 'IFP.TO', 'IGM.TO', 'IMO.TO', 'L.TO', 'LB.TO', 'LNR.TO', 'MATR.TO', 'MRU.TO', 'MX.TO', 'NA.TO', 'ONEX.TO', 'POU.TO', 'POW.TO', 'PRMW.TO', 'QBR-B.TO', 'RCI-B.TO', 'RY.TO', 'SU.TO', 'T.TO', 'TA.TO', 'TD.TO', 'TECK-B.TO', 'TRP.TO', 'WFG.TO', 'WN.TO']  # 48 assets

Components of SSE Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

	SSE =  ['600010.SS', '600028.SS', '600030.SS', '600031.SS', '600048.SS', '600050.SS', '600089.SS', '600104.SS', '600111.SS', '600196.SS', '600276.SS', '600309.SS', '600406.SS', '600436.SS', '600438.SS', '600519.SS', '600690.SS', '600745.SS', '600809.SS', '600887.SS', '600900.SS', '601088.SS', '601166.SS', '601318.SS', '601390.SS', '601398.SS', '601628.SS', '601857.SS', '601899.SS', '601919.SS']  # 30 assets

Components of HSI Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

	HSI =  ['0241.HK', '2020.HK', '3988.HK', '2388.HK', '1211.HK', '0939.HK', '2628.HK', '2319.HK', '3968.HK', '0941.HK', '0688.HK', '0386.HK', '0291.HK', '1109.HK', '0836.HK', '1088.HK', '0762.HK', '0267.HK', '0001.HK', '1038.HK', '0002.HK', '0883.HK', '2007.HK', '1093.HK', '2688.HK', '0027.HK', '0175.HK', '0101.HK', '0011.HK', '0012.HK', '1044.HK', '0003.HK', '0388.HK', '0005.HK', '1398.HK', '0992.HK', '2331.HK', '0823.HK', '0066.HK', '0017.HK', '0316.HK', '0857.HK', '0006.HK', '2313.HK', '0016.HK', '1177.HK', '0981.HK', '2382.HK', '0669.HK', '0700.HK', '0322.HK', '0868.HK', '2899.HK']  # 53 assets

Components of CMEG Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

	CMEG =  ['ZC=F', 'ZS=F', 'ZW=F', 'CT=F', 'CL=F', 'BZ=F', 'NG=F', 'RB=F', 'HO=F', 'ES=F', 'NQ=F', 'YM=F', '6E=F', '6J=F', '6B=F', '6A=F', '6S=F', '6M=F', '6N=F', 'ZT=F', 'ZF=F', 'ZN=F', 'GC=F', 'SI=F', 'HG=F']  # 25 assets

Components of CRYPTO Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

	CRYPTO = ['ADA-USD', 'ANT-USD', 'BAT-USD', 'BCH-USD', 'BNB-USD', 'BTC-USD', 'BTG-USD', 'DASH-USD', 'DCR-USD', 'DGB-USD', 'DOGE-USD', 'ENJ-USD', 'EOS-USD', 'ETC-USD', 'ETH-USD', 'GAS-USD', 'GLM-USD', 'GNO-USD', 'ICX-USD', 'KCS-USD', 'LINK-USD', 'LRC-USD', 'LSK-USD', 'LTC-USD', 'MANA-USD', 'NEO-USD', 'NMR-USD', 'POWR-USD', 'QTUM-USD', 'RLC-USD', 'SC-USD', 'STORJ-USD', 'STRAX-USD', 'TRX-USD', 'USDT-USD', 'WAVES-USD', 'XEM-USD', 'XLM-USD', 'XMR-USD', 'XRP-USD', 'XTZ-USD', 'ZEC-USD', 'ZRX-USD']  # 43 assets











