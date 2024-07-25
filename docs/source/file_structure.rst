File Structure
==============

::

   FinOL
   ├─ LICENSE
   ├─ MANIFEST.in
   ├─ README.md
   ├─ TA_Lib-0.4.24-cp39-cp39-win_amd64.whl
   ├─ figure
   │    ├─ FinOL.png
   │    ├─ logo.png
   │    └─ tutorial_2.png
   ├─ finol
   │    ├─ __init__.py
   │    ├─ config.py
   │    ├─ data
   │    │    ├─ benchmark_results
   │    │    │    ├─ __init__.py
   │    │    │    ├─ other
   │    │    │    │    └─ price_relative
   │    │    │    │           ├─ price_relative_CMEG.mat
   │    │    │    │           ├─ price_relative_CRYPTO.mat
   │    │    │    │           ├─ price_relative_DJIA.mat
   │    │    │    │           ├─ price_relative_HSI.mat
   │    │    │    │           ├─ price_relative_NYSE(N).mat
   │    │    │    │           ├─ price_relative_NYSE(O).mat
   │    │    │    │           ├─ price_relative_SP500.mat
   │    │    │    │           ├─ price_relative_SSE.mat
   │    │    │    │           └─ price_relative_TSE.mat
   │    │    │    ├─ practical_metrics
   │    │    │    │    ├─ CMEG
   │    │    │    │    │    ├─ final_practical_result.xlsx
   │    │    │    │    │    └─ transaction_costs_adjusted_cumulative_wealth.xlsx
   │    │    │    │    ├─ CRYPTO
   │    │    │    │    │    ├─ final_practical_result.xlsx
   │    │    │    │    │    └─ transaction_costs_adjusted_cumulative_wealth.xlsx
   │    │    │    │    ├─ DJIA
   │    │    │    │    │    ├─ final_practical_result.xlsx
   │    │    │    │    │    └─ transaction_costs_adjusted_cumulative_wealth.xlsx
   │    │    │    │    ├─ HSI
   │    │    │    │    │    ├─ final_practical_result.xlsx
   │    │    │    │    │    └─ transaction_costs_adjusted_cumulative_wealth.xlsx
   │    │    │    │    ├─ NYSE(N)
   │    │    │    │    │    ├─ final_practical_result.xlsx
   │    │    │    │    │    └─ transaction_costs_adjusted_cumulative_wealth.xlsx
   │    │    │    │    ├─ NYSE(O)
   │    │    │    │    │    ├─ final_practical_result.xlsx
   │    │    │    │    │    └─ transaction_costs_adjusted_cumulative_wealth.xlsx
   │    │    │    │    ├─ SP500
   │    │    │    │    │    ├─ final_practical_result.xlsx
   │    │    │    │    │    └─ transaction_costs_adjusted_cumulative_wealth.xlsx
   │    │    │    │    ├─ SSE
   │    │    │    │    │    ├─ final_practical_result.xlsx
   │    │    │    │    │    └─ transaction_costs_adjusted_cumulative_wealth.xlsx
   │    │    │    │    └─ TSE
   │    │    │    │           ├─ final_practical_result.xlsx
   │    │    │    │           └─ transaction_costs_adjusted_cumulative_wealth.xlsx
   │    │    │    ├─ profit_metrics
   │    │    │    │    ├─ CMEG
   │    │    │    │    │    ├─ daily_cumulative_wealth.xlsx
   │    │    │    │    │    ├─ daily_return.xlsx
   │    │    │    │    │    └─ final_profit_result.xlsx
   │    │    │    │    ├─ CRYPTO
   │    │    │    │    │    ├─ daily_cumulative_wealth.xlsx
   │    │    │    │    │    ├─ daily_return.xlsx
   │    │    │    │    │    └─ final_profit_result.xlsx
   │    │    │    │    ├─ DJIA
   │    │    │    │    │    ├─ daily_cumulative_wealth.xlsx
   │    │    │    │    │    ├─ daily_return.xlsx
   │    │    │    │    │    └─ final_profit_result.xlsx
   │    │    │    │    ├─ HSI
   │    │    │    │    │    ├─ daily_cumulative_wealth.xlsx
   │    │    │    │    │    ├─ daily_return.xlsx
   │    │    │    │    │    └─ final_profit_result.xlsx
   │    │    │    │    ├─ NYSE(N)
   │    │    │    │    │    ├─ daily_cumulative_wealth.xlsx
   │    │    │    │    │    ├─ daily_return.xlsx
   │    │    │    │    │    └─ final_profit_result.xlsx
   │    │    │    │    ├─ NYSE(O)
   │    │    │    │    │    ├─ daily_cumulative_wealth.xlsx
   │    │    │    │    │    ├─ daily_return.xlsx
   │    │    │    │    │    └─ final_profit_result.xlsx
   │    │    │    │    ├─ SP500
   │    │    │    │    │    ├─ daily_cumulative_wealth.xlsx
   │    │    │    │    │    ├─ daily_return.xlsx
   │    │    │    │    │    └─ final_profit_result.xlsx
   │    │    │    │    ├─ SSE
   │    │    │    │    │    ├─ daily_cumulative_wealth.xlsx
   │    │    │    │    │    ├─ daily_return.xlsx
   │    │    │    │    │    └─ final_profit_result.xlsx
   │    │    │    │    └─ TSE
   │    │    │    │           ├─ daily_cumulative_wealth.xlsx
   │    │    │    │           ├─ daily_return.xlsx
   │    │    │    │           └─ final_profit_result.xlsx
   │    │    │    └─ risk_metrics
   │    │    │           ├─ CMEG
   │    │    │           │    ├─ daily_drawdown.xlsx
   │    │    │           │    └─ final_risk_result.xlsx
   │    │    │           ├─ CRYPTO
   │    │    │           │    ├─ daily_drawdown.xlsx
   │    │    │           │    └─ final_risk_result.xlsx
   │    │    │           ├─ DJIA
   │    │    │           │    ├─ daily_drawdown.xlsx
   │    │    │           │    └─ final_risk_result.xlsx
   │    │    │           ├─ HSI
   │    │    │           │    ├─ daily_drawdown.xlsx
   │    │    │           │    └─ final_risk_result.xlsx
   │    │    │           ├─ NYSE(N)
   │    │    │           │    ├─ daily_drawdown.xlsx
   │    │    │           │    └─ final_risk_result.xlsx
   │    │    │           ├─ NYSE(O)
   │    │    │           │    ├─ daily_drawdown.xlsx
   │    │    │           │    └─ final_risk_result.xlsx
   │    │    │           ├─ SP500
   │    │    │           │    ├─ daily_drawdown.xlsx
   │    │    │           │    └─ final_risk_result.xlsx
   │    │    │           ├─ SSE
   │    │    │           │    ├─ daily_drawdown.xlsx
   │    │    │           │    └─ final_risk_result.xlsx
   │    │    │           └─ TSE
   │    │    │                  ├─ daily_drawdown.xlsx
   │    │    │                  └─ final_risk_result.xlsx
   │    │    └─ datasets
   │    │           ├─ CMEG
   │    │           ├─ CRYPTO
   │    │           ├─ DJIA
   │    │           ├─ HSI
   │    │           ├─ NYSE(N)
   │    │           ├─ NYSE(O)
   │    │           ├─ SP500
   │    │           ├─ SSE
   │    │           └─ TSE
   │    ├─ data_layer
   │    │    ├─ __init__.py
   │    │    ├─ data_loader.py
   │    │    └─ scaler_selector.py
   │    ├─ evaluation_layer
   │    │    ├─ __init__.py
   │    │    ├─ benchmark_loader.py
   │    │    ├─ metric_caculator.py
   │    │    └─ model_evaluator.py
   │    ├─ main
   │    │    ├─ main.ipynb
   │    │    └─ main.py
   │    ├─ model_layer
   │    │    ├─ __init__.py
   │    │    ├─ CNN.py
   │    │    ├─ DNN.py
   │    │    ├─ LSRE_CAAN.py
   │    │    ├─ LSTM.py
   │    │    ├─ RNN.py
   │    │    ├─ Transformer.py
   │    │    └─ model_selector.py
   │    ├─ optimization_layer
   │    │    ├─ __init__.py
   │    │    ├─ criterion_selector.py
   │    │    ├─ model_trainer.py
   │    │    └─ optimizer_selector.py
   │    ├─ setup.py
   │    ├─ tutorials
   │    │    ├─ README.md
   │    │    ├─ _.ipynb
   │    │    ├─ tutorial_1.ipynb
   │    │    ├─ tutorial_2.ipynb
   │    │    ├─ tutorial_3.ipynb
   │    │    └─ tutorial_4.ipynb
   │    ├─ update
   │    │    └─ __init__.py
   │    └─ utils.py
   ├─ logdir
   ├─ requirements.txt
   └─ setup.py
