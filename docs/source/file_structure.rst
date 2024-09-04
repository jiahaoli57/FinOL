File Structure
==============

::

  finol
  ├─ __init__.py
  ├─ config.json
  ├─ APP
  │    ├─ display_info.py
  │    ├─ finol_logo_icon.png
  │    └─ FinOLAPP.py
  ├─ data
  │    ├─ benchmark_results
  │    │    ├─ __init__.py
  │    │    ├─ other
  │    │    │    └─ price_relative
  │    │    │           ├─ price_relative_CMEG.mat
  │    │    │           ├─ ...
  │    │    │           └─ price_relative_TSE.mat
  │    │    ├─ practical_metrics
  │    │    │    ├─ CMEG
  │    │    │    │    ├─ final_practical_result.xlsx
  │    │    │    │    └─ transaction_costs_adjusted_cumulative_wealth.xlsx
  │    │    │    ├─ ...
  │    │    │    │    ├─ final_practical_result.xlsx
  │    │    │    │    └─ transaction_costs_adjusted_cumulative_wealth.xlsx
  │    │    │    └─ TSE
  │    │    │           ├─ final_practical_result.xlsx
  │    │    │           └─ transaction_costs_adjusted_cumulative_wealth.xlsx
  │    │    ├─ profit_metrics
  │    │    │    ├─ CMEG
  │    │    │    │    ├─ daily_cumulative_wealth.xlsx
  │    │    │    │    ├─ daily_return.xlsx
  │    │    │    │    └─ final_profit_result.xlsx
  │    │    │    ├─ ...
  │    │    │    │    ├─ daily_cumulative_wealth.xlsx
  │    │    │    │    ├─ daily_return.xlsx
  │    │    │    │    └─ final_profit_result.xlsx
  │    │    │    └─ TSE
  │    │    │           ├─ daily_cumulative_wealth.xlsx
  │    │    │           ├─ daily_return.xlsx
  │    │    │           └─ final_profit_result.xlsx
  │    │    └─ risk_metrics
  │    │           ├─ CMEG
  │    │           │    ├─ daily_drawdown.xlsx
  │    │           │    └─ final_risk_result.xlsx
  │    │           ├─ ...
  │    │           │    ├─ daily_drawdown.xlsx
  │    │           │    └─ final_risk_result.xlsx
  │    │           └─ TSE
  │    │                  ├─ daily_drawdown.xlsx
  │    │                  └─ final_risk_result.xlsx
  │    └─ datasets
  │           ├─ CMEG
  │           │    ├─ 6A=F.xlsx
  │           │    ├─ ...
  │           │    └─ ZW=F.xlsx
  │           ├─ ...
  │           │    ├─ ...
  │           │    ├─ ...
  │           │    └─ ...
  │           └─ TSE
  │                  ├─ ABX.TO.xlsx
  │                  ├─ ...
  │                  └─ WN.TO.xlsx
  ├─ data_layer
  │    ├─ __init__.py
  │    ├─ dataset_loader.py
  │    └─ scaler_selector.py
  ├─ evaluation_layer
  │    ├─ __init__.py
  │    ├─ benchmark_loader.py
  │    ├─ distiller_selector.py
  │    ├─ economic_distiller.py
  │    ├─ metric_caculator.py
  │    ├─ model_evaluator.py
  │    └─ result_visualizer.py
  ├─ logdir
  │    ├─ 2024-05-20_15-25-17
  │    ├─ ...
  │    └─ 2024-05-20_15-28-32
  ├─ main
  │    ├─ main.ipynb
  │    └─ main.py
  ├─ model_layer
  │    ├─ __init__.py
  │    ├─ AlphaPortfolio.py
  │    ├─ CNN.py
  │    ├─ DNN.py
  │    ├─ LSRE_CAAN.py
  │    ├─ LSTM.py
  │    ├─ RNN.py
  │    ├─ Transformer.py
  │    ├─ CustomModel.py
  │    └─ model_instantiator.py
  ├─ optimization_layer
  │    ├─ __init__.py
  │    ├─ criterion_selector.py
  │    ├─ model_trainer.py
  │    ├─ optimizer_selector.py
  │    └─ parameters_tuner.py
  ├─ tutorials
  │    ├─ README.md
  │    └─ tutorial_quickstart.ipynb
  ├─ update
  │    └─ __init__.py
  └─ utils.py
