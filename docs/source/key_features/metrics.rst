.. _supported_metrics:

Comprehensive Metrics
===================================================

.. table::
   :class: ghost

   +----------------------------------------------+------------------+------------------------------------------------------------------------------------+
   | Name                                         | Abbreviation     | Function Call                                                                      |
   +==============================================+==================+====================================================================================+
   | *Profit Metric*                                                                                                                                      |
   +----------------------------------------------+------------------+------------------------------------------------------------------------------------+
   | Cumulative Wealth                            | CW               | :func:`~finol.evaluation_layer.MetricCalculator.calculate_final_cumulative_wealth` |
   +----------------------------------------------+------------------+------------------------------------------------------------------------------------+
   | Annualized Percentage Yield                  | APY              | :func:`~finol.evaluation_layer.MetricCalculator.calculate_annual_percentage_yield` |
   +----------------------------------------------+------------------+------------------------------------------------------------------------------------+
   | Sharpe Ratio                                 | SR               | :func:`~finol.evaluation_layer.MetricCalculator.calculate_sharpe_ratio`            |
   +----------------------------------------------+------------------+------------------------------------------------------------------------------------+
   | *Risk Metric*                                                                                                                                        |
   +----------------------------------------------+------------------+------------------------------------------------------------------------------------+
   | Volatility Risk                              | VR               | :func:`~finol.evaluation_layer.MetricCalculator.calculate_volatility_risk`         |
   +----------------------------------------------+------------------+------------------------------------------------------------------------------------+
   | Maximum DrawDown                             | MDD              | :func:`~finol.evaluation_layer.MetricCalculator.calculate_maximum_drawdown`        |
   +----------------------------------------------+------------------+------------------------------------------------------------------------------------+
   | *Practical Metric*                                                                                                                                   |
   +----------------------------------------------+------------------+------------------------------------------------------------------------------------+
   | Average Turnover                             | ATO              | :func:`~finol.evaluation_layer.MetricCalculator.calculate_average_turnover`        |
   +----------------------------------------------+------------------+------------------------------------------------------------------------------------+
   | Transaction Costs-Adjusted Cumulative Wealth | TCW              | :func:`~finol.evaluation_layer.MetricCalculator.calculate_transaction_costs_wealth`|
   +----------------------------------------------+------------------+------------------------------------------------------------------------------------+
   | Running Time                                 | RT               | :func:`~finol.evaluation_layer.MetricCalculator.calculate_runtime`                 |
   +----------------------------------------------+------------------+------------------------------------------------------------------------------------+

.. role:: hidden
   :class: hidden

.. directive:: custom-comment
   :hidden:

.. custom-comment::
   :hidden:

    Profitability
    ~~~~~~~~~~~~~

    As of July 2024, the top five baselines for profitability (quantified by the size of CW metric) are:

    .. list-table::
       :widths: 20, 2, 2, 2, 2, 2, 2, 2, 2, 2
       :header-rows: 1
       :class: ghost

       * - Ranking
         - NYSE(O)
         - NYSE(N)
         - DJIA
         - SP500
         - TSE
         - SSE
         - HSI
         - CMEG
         - CRYPTO
       * - |:1st_place_medal:| **1st**
         - ANTI\ :sup:`2`\
         - SSPO
         - SSPO
         - SSPO
         - RPRT
         - BCRP
         - BCRP
         - BCRP
         - BCRP
       * - |:2nd_place_medal:| **2nd**
         - ANTI\ :sup:`1`\
         - BCRP
         - PPT
         - PPT
         - PPT
         - Best
         - Best
         - Best
         - Best
       * - |:3rd_place_medal:| **3rd**
         - ONS
         - ANTI\ :sup:`2`\
         - KTPT
         - KTPT
         - SSPO
         - GRW
         - SSPO
         - SSPO
         - ONS
       * - |:reminder_ribbon:| **4th**
         - BCRP
         - PPT
         - PAMR
         - ANTI\ :sup:`1`\
         - RMR
         - ANTI\ :sup:`2`\
         - GRW
         - PPT
         - ANTI\ :sup:`2`\
       * - |:reminder_ribbon:| **5th**
         - PPT
         - PAMR
         - CWMR-Stdev
         - ANTI\ :sup:`2`\
         - AICTR
         - ANTI\ :sup:`1`\
         - CWMR-Stdev
         - ANTI\ :sup:`2`\
         - GRW

    Risk Resilience
    ~~~~~~~~~~~~~~~

    As of July 2024, the top five baselines for risk resilience (quantified by the size of MDD metric) are:

    .. list-table::
       :widths: 2, 2, 2, 2, 2, 2, 2, 2, 2, 2
       :header-rows: 1
       :class: ghost

       * - Ranking
         - NYSE(O)
         - NYSE(N)
         - DJIA
         - SP500
         - TSE
         - SSE
         - HSI
         - CMEG
         - CRYPTO
       * - |:1st_place_medal:| **1st**
         - BCRP
         - GRW
         - KTPT
         - Market
         - ANTI\ :sup:`2`\
         - UP
         - WAAS
         - ONS
         - ANTI\ :sup:`2`\
       * - |:2nd_place_medal:| **2nd**
         - Best
         - EG
         - PPT
         - Best
         - ANTI\ :sup:`1`\
         - UCRP
         - Market
         - GRW
         - ANTI\ :sup:`1`\
       * - |:3rd_place_medal:| **3rd**
         - ANTI\ :sup:`1`\
         - WAAS
         - SSPO
         - UCRP
         - RMR
         - SP
         - EG
         - SP
         - BCRP
       * - |:reminder_ribbon:| **4th**
         - UCRP
         - SP
         - GRW
         - BCRP
         - OLMAR-S
         - EG
         - UCRP
         - UCRP
         - ONS
       * - |:reminder_ribbon:| **5th**
         - SP
         - UCRP
         - Best
         - UP
         - PPT
         - WAAS
         - SP
         - UP
         - SP

    Practicality
    ~~~~~~~~~~~~

    As of July 2024, the top five baselines for practicality (quantified by the size of ATO metric) are:


    .. list-table::
       :widths: 2, 2, 2, 2, 2, 2, 2, 2, 2, 2
       :header-rows: 1
       :class: ghost

       * - Ranking
         - NYSE(O)
         - NYSE(N)
         - DJIA
         - SP500
         - TSE
         - SSE
         - HSI
         - CMEG
         - CRYPTO
       * - |:1st_place_medal:| **1st**
         - BCRP
         - EG
         - BCRP
         - BCRP
         - BCRP
         - BCRP
         - EG
         - BCRP
         - EG
       * - |:2nd_place_medal:| **2nd**
         - EG
         - UP
         - EG
         - EG
         - EG
         - EG
         - UCRP
         - EG
         - UCRP
       * - |:3rd_place_medal:| **3rd**
         - CW-OGD
         - UCRP
         - SP
         - SP
         - UCRP
         - UCRP
         - SP
         - SP
         - SP
       * - |:reminder_ribbon:| **4th**
         - GRW
         - SP
         - UCRP
         - UCRP
         - SP
         - SP
         - WAAS
         - UCRP
         - WAAS
       * - |:reminder_ribbon:| **5th**
         - UCRP
         - WAAS
         - WAAS
         - UP
         - WAAS
         - WAAS
         - BCRP
         - WAAS
         - BCRP
