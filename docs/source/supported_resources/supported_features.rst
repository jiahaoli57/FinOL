.. _supported_features:

FinOL Features
========

This section provides detailed definitions for the key data features in ``FinOL``.
These features can help describe and analyze financial time series from multiple perspectives.
The following tables categorize these features into 8 main groups and list the name, function call and description
for each feature. For the detailed computation process, please refer to
:func:`~finol.data_layer.DatasetLoader.feature_engineering`.


.. contents::
    :local:

.. _OHLCV_features:

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

.. _overlap_features:

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

.. _momentum_features:

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

.. _volume_features:

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

.. _cycle_features:

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

.. _price_features:

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

.. _volatility_features:

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

.. _pattern_features:

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
   * - CDL3BLACKCROWS
     - ``ta.CDL3BLACKCROWS(df.OPEN, df.HIGH, df.LOW, df.CLOSE)``
     - Three Black Crows
   * - CDL3INSIDE
     - ``ta.CDL3INSIDE(df.OPEN, df.HIGH, df.LOW, df.CLOSE)``
     - Three Inside Up/Down
   * - CDL3LINESTRIKE
     - ``ta.CDL3LINESTRIKE(df.OPEN, df.HIGH, df.LOW, df.CLOSE)``
     - Three-Line Strike
   * - CDL3OUTSIDE
     - ``ta.CDL3OUTSIDE(df.OPEN, df.HIGH, df.LOW, df.CLOSE)``
     - Three Outside Up/Down
   * - CDL3STARSINSOUTH
     - ``ta.CDL3STARSINSOUTH(df.OPEN, df.HIGH, df.LOW, df.CLOSE)``
     - Three Stars In The South
   * - CDL3WHITESOLDIERS
     - ``ta.CDL3WHITESOLDIERS(df.OPEN, df.HIGH, df.LOW, df.CLOSE)``
     - Three Advancing White Soldiers
   * - CDLABANDONEDBABY
     - ``ta.CDLABANDONEDBABY(df.OPEN, df.HIGH, df.LOW, df.CLOSE)``
     - Abandoned Baby
   * - CDLADVANCEBLOCK
     - ``ta.CDLADVANCEBLOCK(df.OPEN, df.HIGH, df.LOW, df.CLOSE)``
     - Advance Block
   * - CDLBELTHOLD
     - ``ta.CDLBELTHOLD(df.OPEN, df.HIGH, df.LOW, df.CLOSE)``
     - Belt-Hold
   * - CDLBREAKAWAY
     - ``ta.CDLBREAKAWAY(df.OPEN, df.HIGH, df.LOW, df.CLOSE)``
     - Breakaway
   * - CDLCLOSINGMARUBOZU
     - ``ta.CDLCLOSINGMARUBOZU(df.OPEN, df.HIGH, df.LOW, df.CLOSE)``
     - Closing Marubozu
   * - CDLCONCEALBABYSWALL
     - ``ta.CDLCONCEALBABYSWALL(df.OPEN, df.HIGH, df.LOW, df.CLOSE)``
     - Concealing Baby Swallow
   * - CDLCOUNTERATTACK
     - ``ta.CDLCOUNTERATTACK(df.OPEN, df.HIGH, df.LOW, df.CLOSE)``
     - Counterattack
   * - CDLDARKCLOUDCOVER
     - ``ta.CDLDARKCLOUDCOVER(df.OPEN, df.HIGH, df.LOW, df.CLOSE)``
     - Dark Cloud Cover
   * - CDLDOJI
     - ``ta.CDLDOJI(df.OPEN, df.HIGH, df.LOW, df.CLOSE)``
     - Doji
   * - CDLDOJISTAR
     - ``ta.CDLDOJISTAR(df.OPEN, df.HIGH, df.LOW, df.CLOSE)``
     - Doji Star
   * - CDLDRAGONFLYDOJI
     - ``ta.CDLDRAGONFLYDOJI(df.OPEN, df.HIGH, df.LOW, df.CLOSE)``
     - Dragonfly Doji
   * - CDLENGULFING
     - ``ta.CDLENGULFING(df.OPEN, df.HIGH, df.LOW, df.CLOSE)``
     - Engulfing Pattern
   * - CDLEVENINGDOJISTAR
     - ``ta.CDLEVENINGDOJISTAR(df.OPEN, df.HIGH, df.LOW, df.CLOSE)``
     - Evening Doji Star
   * - CDLEVENINGSTAR
     - ``ta.CDLEVENINGSTAR(df.OPEN, df.HIGH, df.LOW, df.CLOSE)``
     - Evening Star
   * - CDLGAPSIDESIDEWHITE
     - ``ta.CDLGAPSIDESIDEWHITE(df.OPEN, df.HIGH, df.LOW, df.CLOSE)``
     - Up/Down-Gap Side-By-Side White Lines
   * - CDLGRAVESTONEDOJI
     - ``ta.CDLGRAVESTONEDOJI(df.OPEN, df.HIGH, df.LOW, df.CLOSE)``
     - Gravestone Doji
   * - CDLHAMMER
     - ``ta.CDLHAMMER(df.OPEN, df.HIGH, df.LOW, df.CLOSE)``
     - Hammer
   * - CDLHANGINGMAN
     - ``ta.CDLHANGINGMAN(df.OPEN, df.HIGH, df.LOW, df.CLOSE)``
     - Hanging Man
   * - CDLHARAMI
     - ``ta.CDLHARAMI(df.OPEN, df.HIGH, df.LOW, df.CLOSE)``
     - Harami Pattern
   * - CDLHARAMICROSS
     - ``ta.CDLHARAMICROSS(df.OPEN, df.HIGH, df.LOW, df.CLOSE)``
     - Harami Cross Pattern
   * - CDLHIGHWAVE
     - ``ta.CDLHIGHWAVE(df.OPEN, df.HIGH, df.LOW, df.CLOSE)``
     - High-Wave Candle
   * - CDLHIKKAKE
     - ``ta.CDLHIKKAKE(df.OPEN, df.HIGH, df.LOW, df.CLOSE)``
     - Hikkake Pattern
   * - CDLHIKKAKEMOD
     - ``ta.CDLHIKKAKEMOD(df.OPEN, df.HIGH, df.LOW, df.CLOSE)``
     - Modified Hikkake Pattern
   * - CDLHOMINGPIGEON
     - ``ta.CDLHOMINGPIGEON(df.OPEN, df.HIGH, df.LOW, df.CLOSE)``
     - Homing Pigeon
   * - CDLIDENTICAL3CROWS
     - ``ta.CDLIDENTICAL3CROWS(df.OPEN, df.HIGH, df.LOW, df.CLOSE)``
     - Identical Three Crows
   * - CDLINNECK
     - ``ta.CDLINNECK(df.OPEN, df.HIGH, df.LOW, df.CLOSE)``
     - In-Neck Pattern
   * - CDLINVERTEDHAMMER
     - ``ta.CDLINVERTEDHAMMER(df.OPEN, df.HIGH, df.LOW, df.CLOSE)``
     - Inverted Hammer
   * - CDLKICKING
     - ``ta.CDLKICKING(df.OPEN, df.HIGH, df.LOW, df.CLOSE)``
     - Kicking
   * - CDLKICKINGBYLENGTH
     - ``ta.CDLKICKINGBYLENGTH(df.OPEN, df.HIGH, df.LOW, df.CLOSE)``
     - Kicking - Bull/Bear Determined by the Longer Marubozu
   * - CDLLADDERBOTTOM
     - ``ta.CDLLADDERBOTTOM(df.OPEN, df.HIGH, df.LOW, df.CLOSE)``
     - Ladder Bottom
   * - CDLLONGLEGGEDDOJI
     - ``ta.CDLLONGLEGGEDDOJI(df.OPEN, df.HIGH, df.LOW, df.CLOSE)``
     - Long Legged Doji
   * - CDLLONGLINE
     - ``ta.CDLLONGLINE(df.OPEN, df.HIGH, df.LOW, df.CLOSE)``
     - Long Line Candle
   * - CDLMARUBOZU
     - ``ta.CDLMARUBOZU(df.OPEN, df.HIGH, df.LOW, df.CLOSE)``
     - Marubozu
   * - CDLMATCHINGLOW
     - ``ta.CDLMATCHINGLOW(df.OPEN, df.HIGH, df.LOW, df.CLOSE)``
     - Matching Low
   * - CDLMATHOLD
     - ``ta.CDLMATHOLD(df.OPEN, df.HIGH, df.LOW, df.CLOSE)``
     - Mat Hold
   * - CDLMORNINGDOJISTAR
     - ``ta.CDLMORNINGDOJISTAR(df.OPEN, df.HIGH, df.LOW, df.CLOSE)``
     - Morning Doji Star
   * - CDLMORNINGSTAR
     - ``ta.CDLMORNINGSTAR(df.OPEN, df.HIGH, df.LOW, df.CLOSE)``
     - Morning Star
   * - CDLONNECK
     - ``ta.CDLONNECK(df.OPEN, df.HIGH, df.LOW, df.CLOSE)``
     - On-Neck Pattern
   * - CDLPIERCING
     - ``ta.CDLPIERCING(df.OPEN, df.HIGH, df.LOW, df.CLOSE)``
     - Piercing Pattern
   * - CDLRICKSHAWMAN
     - ``ta.CDLRICKSHAWMAN(df.OPEN, df.HIGH, df.LOW, df.CLOSE)``
     - Rickshaw Man
   * - CDLRISEFALL3METHODS
     - ``ta.CDLRISEFALL3METHODS(df.OPEN, df.HIGH, df.LOW, df.CLOSE)``
     - Rising/Falling Three Methods
   * - CDLSEPARATINGLINES
     - ``ta.CDLSEPARATINGLINES(df.OPEN, df.HIGH, df.LOW, df.CLOSE)``
     - Separating Lines
   * - CDLSHOOTINGSTAR
     - ``ta.CDLSHOOTINGSTAR(df.OPEN, df.HIGH, df.LOW, df.CLOSE)``
     - Shooting Star
   * - CDLSHORTLINE
     - ``ta.CDLSHORTLINE(df.OPEN, df.HIGH, df.LOW, df.CLOSE)``
     - Short Line Candle
   * - CDLSPINNINGTOP
     - ``ta.CDLSPINNINGTOP(df.OPEN, df.HIGH, df.LOW, df.CLOSE)``
     - Spinning Top
   * - CDLSTALLEDPATTERN
     - ``ta.CDLSTALLEDPATTERN(df.OPEN, df.HIGH, df.LOW, df.CLOSE)``
     - Stalled Pattern
   * - CDLSTICKSANDWICH
     - ``ta.CDLSTICKSANDWICH(df.OPEN, df.HIGH, df.LOW, df.CLOSE)``
     - Stick Sandwich
   * - CDLTAKURI
     - ``ta.CDLTAKURI(df.OPEN, df.HIGH, df.LOW, df.CLOSE)``
     - Takuri (Dragonfly Doji with Very Long Lower Shadow)
   * - CDLTASUKIGAP
     - ``ta.CDLTASUKIGAP(df.OPEN, df.HIGH, df.LOW, df.CLOSE)``
     - Tasuki Gap
   * - CDLTHRUSTING
     - ``ta.CDLTHRUSTING(df.OPEN, df.HIGH, df.LOW, df.CLOSE)``
     - Thrusting Pattern
   * - CDLTRISTAR
     - ``ta.CDLTRISTAR(df.OPEN, df.HIGH, df.LOW, df.CLOSE)``
     - Tristar Pattern
   * - CDLUNIQUE3RIVER
     - ``ta.CDLUNIQUE3RIVER(df.OPEN, df.HIGH, df.LOW, df.CLOSE)``
     - Unique 3 River
   * - CDLUPSIDEGAP2CROWS
     - ``ta.CDLUPSIDEGAP2CROWS(df.OPEN, df.HIGH, df.LOW, df.CLOSE)``
     - Upside Gap Two Crows
   * - CDLXSIDEGAP3METHODS
     - ``ta.CDLXSIDEGAP3METHODS(df.OPEN, df.HIGH, df.LOW, df.CLOSE)``
     - Upside/Downside Gap Three Methods
