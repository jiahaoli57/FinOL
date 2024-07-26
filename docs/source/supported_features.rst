.. _supported_features:

Supported Features
==================

.. contents::
    :local:

OHLCV Features
--------------

.. list-table::
   :header-rows: 1

   * - Feature Name
     - Function Call
     - Description
   * - OPEN
     - ``df.OPEN``
     - Open Price
   * - HIGH
     - ``df.HIGH``
     - High Price
   * - LOW
     - ``df.LOW``
     - Low Price
   * - CLOSE
     - ``df.CLOSE``
     - Close Price
   * - VOLUME
     - ``df.VOLUME``
     - Trading Volume


Overlap Features
----------------

.. list-table::
   :header-rows: 1

   * - Feature Name
     - Function Call
     - Description
   * - BBANDS_UPPER
     - ``ta.BBANDS(df.CLOSE)[0]``
     - Bollinger Bands - Upper Band
   * - BBANDS_MIDDLE
     - ``ta.BBANDS(df.CLOSE)[1]``
     - Bollinger Bands - Middle Band
   * - BBANDS_LOWER
     - ``ta.BBANDS(df.CLOSE)[2]``
     - Bollinger Bands - Lower Band
   * - DEMA
     - ``ta.DEMA(df.CLOSE)``
     - Double Exponential Moving Average
   * - EMA
     - ``ta.EMA(df.CLOSE)``
     - Exponential Moving Average
   * - HT_TRENDLINE
     - ``ta.HT_TRENDLINE(df.CLOSE)``
     - Hilbert Transform - Instantaneous Trendline
   * - KAMA
     - ``ta.KAMA(df.CLOSE)``
     - Kaufman Adaptive Moving Average
   * - MA
     - ``ta.MA(df.CLOSE)``
     - Moving Average
   * - MAMA
     - ``ta.MAMA(df.CLOSE)[0]``
     - MESA Adaptive Moving Average - MAMA
   * - MAMA_FAMA
     - ``ta.MAMA(df.CLOSE)[1]``
     - MESA Adaptive Moving Average - FAMA
   * - MAVP
     - ``ta.MAVP(df.CLOSE, df.DATE)``
     - Moving Average with Variable Period
   * - MIDPOINT
     - ``ta.MIDPOINT(df.CLOSE)``
     - MidPoint over Period
   * - MIDPRICE
     - ``ta.MIDPRICE(df.HIGH, df.LOW)``
     - Midpoint Price over Period
   * - SAR
     - ``ta.SAR(df.HIGH, df.LOW)``
     - Parabolic SAR
   * - SAREXT
     - ``ta.SAREXT(df.HIGH, df.LOW)``
     - Parabolic SAR - Extended
   * - SMA
     - ``ta.SMA(df.CLOSE)``
     - Simple Moving Average
   * - T3
     - ``ta.T3(df.CLOSE)``
     - Triple Exponential Moving Average (T3)
   * - TEMA
     - ``ta.TEMA(df.CLOSE)``
     - Triple Exponential Moving Average
   * - TRIMA
     - ``ta.TRIMA(df.CLOSE)``
     - Triangular Moving Average
   * - WMA
     - ``ta.WMA(df.CLOSE)``
     - Weighted Moving Average


Momentum Features
-----------------

.. list-table::
   :header-rows: 1

   * - Feature Name
     - Function Call
     - Description
   * - ADX
     - ``ta.ADX(df.HIGH, df.LOW, df.CLOSE)``
     - Average Directional Movement Index
   * - ADXR
     - ``ta.ADXR(df.HIGH, df.LOW, df.CLOSE)``
     - Average Directional Movement Index Rating
   * - APO
     - ``ta.APO(df.CLOSE)``
     - Absolute Price Oscillator
   * - AROON_UP
     - ``ta.AROON(df.HIGH, df.LOW)[0]``
     - Aroon Up
   * - AROON_DOWN
     - ``ta.AROON(df.HIGH, df.LOW)[1]``
     - Aroon Down
   * - AROONOSC
     - ``ta.AROONOSC(df.HIGH, df.LOW)``
     - Aroon Oscillator
   * - BOP
     - ``ta.BOP(df.OPEN, df.HIGH, df.LOW, df.CLOSE)``
     - Balance Of Power
   * - CCI
     - ``ta.CCI(df.HIGH, df.LOW, df.CLOSE)``
     - Commodity Channel Index
   * - CMO
     - ``ta.CMO(df.CLOSE)``
     - Chande Momentum Oscillator
   * - DX
     - ``ta.DX(df.HIGH, df.LOW, df.CLOSE)``
     - Directional Movement Index
   * - MACD
     - ``ta.MACD(df.CLOSE)[0]``
     - Moving Average Convergence/Divergence
   * - MACD_SIGNAL
     - ``ta.MACD(df.CLOSE)[1]``
     - MACD Signal Line
   * - MACD_HIST
     - ``ta.MACD(df.CLOSE)[2]``
     - MACD Histogram
   * - MACDEXT
     - ``ta.MACDEXT(df.CLOSE)[0]``
     - MACD with controllable MA type
   * - MACDEXT_SIGNAL
     - ``ta.MACDEXT(df.CLOSE)[1]``
     - MACDEXT Signal Line
   * - MACDEXT_HIST
     - ``ta.MACDEXT(df.CLOSE)[2]``
     - MACDEXT Histogram
   * - MACDFIX
     - ``ta.MACDFIX(df.CLOSE)[0]``
     - Moving Average Convergence/Divergence Fix 12/26
   * - MACDFIX_SIGNAL
     - ``ta.MACDFIX(df.CLOSE)[1]``
     - MACDFIX Signal Line
   * - MACDFIX_HIST
     - ``ta.MACDFIX(df.CLOSE)[2]``
     - MACDFIX Histogram
   * - MFI
     - ``ta.MFI(df.HIGH, df.LOW, df.CLOSE, df.VOLUME)``
     - Money Flow Index
   * - MINUS_DI
     - ``ta.MINUS_DI(df.HIGH, df.LOW, df.CLOSE)``
     - Minus Directional Indicator
   * - MINUS_DM
     - ``ta.MINUS_DM(df.HIGH, df.LOW)``
     - Minus Directional Movement
   * - MOM
     - ``ta.MOM(df.CLOSE)``
     - Momentum
   * - PLUS_DI
     - ``ta.PLUS_DI(df.HIGH, df.LOW, df.CLOSE)``
     - Plus Directional Indicator
   * - PLUS_DM
     - ``ta.PLUS_DM(df.HIGH, df.LOW)``
     - Plus Directional Movement
   * - PPO
     - ``ta.PPO(df.CLOSE)``
     - Percentage Price Oscillator
   * - ROC
     - ``ta.ROC(df.CLOSE)``
     - Rate of change: ((price/prevPrice)-1)*100
   * - ROCP
     - ``ta.ROCP(df.CLOSE)``
     - Rate of change Percentage: (price-prevPrice)/prevPrice
   * - ROCR
     - ``ta.ROCR(df.CLOSE)``
     - Rate of change ratio: (price/prevPrice)
   * - ROCR100
     - ``ta.ROCR100(df.CLOSE)``
     - Rate of change ratio 100 scale: (price/prevPrice)*100
   * - RSI
     - ``ta.RSI(df.CLOSE)``
     - Relative Strength Index
   * - STOCH_K
     - ``ta.STOCH(df.HIGH, df.LOW, df.CLOSE)[0]``
     - Stochastic %K
   * - STOCH_D
     - ``ta.STOCH(df.HIGH, df.LOW, df.CLOSE)[1]``
     - Stochastic %D
   * - STOCHF_K
     - ``ta.STOCHF(df.HIGH, df.LOW, df.CLOSE)[0]``
     - Stochastic Fast %K
   * - STOCHF_D
     - ``ta.STOCHF(df.HIGH, df.LOW, df.CLOSE)[1]``
     - Stochastic Fast %D
   * - STOCHRSI_K
     - ``ta.STOCHRSI(df.CLOSE)[0]``
     - Stochastic RSI %K
   * - STOCHRSI_D
     - ``ta.STOCHRSI(df.CLOSE)[1]``
     - Stochastic RSI %D
   * - TRIX
     - ``ta.TRIX(df.CLOSE)``
     - 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA
   * - ULTOSC
     - ``ta.ULTOSC(df.HIGH, df.LOW, df.CLOSE)``
     - Ultimate Oscillator
   * - WILLR
     - ``ta.WILLR(df.HIGH, df.LOW, df.CLOSE)``
     - Williams' %R

Volume Features
---------------

.. list-table::
   :header-rows: 1

   * - Feature Name
     - Function Call
     - Description
   * - AD
     - ``ta.AD(df.HIGH, df.LOW, df.CLOSE, df.VOLUME)``
     - Chaikin A/D Line
   * - ADOSC
     - ``ta.ADOSC(df.HIGH, df.LOW, df.CLOSE, df.VOLUME)``
     - Chaikin A/D Oscillator
   * - OBV
     - ``ta.OBV(df.CLOSE, df.VOLUME)``
     - On Balance Volume

Cycle Features
--------------

.. list-table::
   :header-rows: 1

   * - Feature Name
     - Function Call
     - Description
   * - HT_DCPERIOD
     - ``ta.HT_DCPERIOD(df.CLOSE)``
     - Hilbert Transform - Dominant Cycle Period
   * - HT_DCPHASE
     - ``ta.HT_DCPHASE(df.CLOSE)``
     - Hilbert Transform - Dominant Cycle Phase
   * - HT_PHASOR_INPHASE
     - ``ta.HT_PHASOR(df.CLOSE)[0]``
     - Hilbert Transform - Phasor Components, In-Phase Component
   * - HT_PHASOR_QUADRATURE
     - ``ta.HT_PHASOR(df.CLOSE)[1]``
     - Hilbert Transform - Phasor Components, Quadrature Component
   * - HT_SINE_LEADSINE
     - ``ta.HT_SINE(df.CLOSE)[0]``
     - Hilbert Transform - SineWave, Lead SineWave
   * - HT_SINE_SINEWAVE
     - ``ta.HT_SINE(df.CLOSE)[1]``
     - Hilbert Transform - SineWave, SineWave
   * - HT_TRENDMODE
     - ``ta.HT_TRENDMODE(df.CLOSE)``
     - Hilbert Transform - Trend vs Cycle Mode

Price Features
--------------

.. list-table::
   :header-rows: 1

   * - Feature Name
     - Function Call
     - Description
   * - AVGPRICE
     - ``ta.AVGPRICE(df.OPEN, df.HIGH, df.LOW, df.CLOSE)``
     - Average Price
   * - MEDPRICE
     - ``ta.MEDPRICE(df.HIGH, df.LOW)``
     - Median Price
   * - TYPPRICE
     - ``ta.TYPPRICE(df.HIGH, df.LOW, df.CLOSE)``
     - Typical Price
   * - WCLPRICE
     - ``ta.WCLPRICE(df.HIGH, df.LOW, df.CLOSE)``
     - Weighted Close Price

Volatility Features
-------------------

.. list-table::
   :header-rows: 1

   * - Feature Name
     - Function Call
     - Description
   * - ATR
     - ``ta.ATR(df.HIGH, df.LOW, df.CLOSE)``
     - Average True Range
   * - NATR
     - ``ta.NATR(df.HIGH, df.LOW, df.CLOSE)``
     - Normalized Average True Range
   * - TRANGE
     - ``ta.TRANGE(df.HIGH, df.LOW, df.CLOSE)``
     - True Range

Pattern Features
----------------

.. list-table::
   :header-rows: 1

   * - Feature Name
     - Function Call
     - Description
   * - CDL2CROWS
     - ``ta.CDL2CROWS(df.OPEN, df.HIGH, df.LOW, df.CLOSE)``
     - Two Crows