# FinOL: Towards Open Benchmarking for Data-Driven Online Portfolio Selection

<div align="center">
<img align="center" src=figure/logo.png width="20%"/> 

<div>&nbsp;</div>

[![Python 3.9](https://shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-3916/)
[![Platform](https://img.shields.io/badge/platform-linux%20%7C%20windows%20%7C%20macos-lightgrey)](Platform)
[![License](https://img.shields.io/github/license/jiahaoli57/FinOL)](LICENSE)
[![PyPI Downloads](https://img.shields.io/pypi/dm/finol)](https://pypi.org/project/finol)
[![Discord](https://img.shields.io/discord/1201132123722104902)](https://discord.gg/KCXQt7r3)

[//]: # ([![Document]&#40;https://img.shields.io/badge/docs-latest-red&#41;]&#40;https://finol.readthedocs.io/en/latest/&#41;)
[//]: # ([![]&#40;https://dcbadge.vercel.app/api/server/KCXQt7r3&#41;]&#40;https://discord.gg/KCXQt7r3&#41;)
[//]: # ([![GitHub stars]&#40;https://img.shields.io/github/stars/ai4finol/finol?color=orange&#41;]&#40;https://github.com/ai4finol/finol/stargazers&#41;)

</div>

***
``FinOL`` represents a pioneering open database for facilitating data-driven financial research. As an
ambitious project, it collects and organizes extensive assets from global markets over half a century,
it provides a long-awaited unified platform to advance data-driven OLPS research.

## :star: **What's NEW!** 

<div align="center">

| Update                                                                               | Status                      |
|--------------------------------------------------------------------------------------|-----------------------------|
| Release  ``FinOL`` [tutorials](finol/tutorials)                                      | Released on 22 March 2024   |
| Release  ``FinOL`` [v0.0.1](https://github.com/jiahaoli57/finol/releases/tag/v0.0.1) | Released on 21 March 2024   |

</div>


## Outline

- [FinOL: Towards Open Benchmarking for Data-Driven Online Portfolio Selection](#finol)
  - [Outline](#outline)
  - [About](#about)
  - [Why should I use FinOL?](#why-should-i-use-finOL?)
  - [Installation](#installation)
  - [Examples and Tutorials](#examples-and-tutorials)
  - [Using FinOL](#using-FinOL)
  - [File Structure](#file-structure)
  - [Contact Us](#contact-us)

## About

Online portfolio selection (OLPS) is an important issue in operations research community that studies how to dynamically
adjust portfolios according to market changes. In the past, OLPS research relied on a general database called ``OLPS`` 
containing price relatives data of financial assets across different markets. However, with the widespread adoption of 
data-driven technologies like machine learning in finance, ``OLPS`` can no longer meet the needs of OLPS research due 
to the lack of support for high-dimensional feature spaces. To solve 
this problem, we propose ``FinOL``, an open financial platform for advancing research in data-driven OLPS. ``FinOL`` expands 
and enriches the previous ``OLPS`` database, containing 9 benchmark financial datasets from 1962 to present across global 
markets. To promote fair comparisons, we evaluate a large number of past classic OLPS methods on ``FinOL``, providing 
reusable benchmark results for future ``FinOL`` users and effectively supporting OLPS research. More importantly, to 
lower the barriers to research, ``FinOL`` provides a complete data-training-testing suite with just three lines of 
command. We are also committed to regularly updating ``FinOL`` with new data and benchmark results reflecting the latest 
developments and trends in the field. This ensures ``FinOL`` remains a valuable resource as data-driven OLPS methods 
continue evolving.

<p align="center">
<img src="figure/FinOL.png" alt>
<em>Overall Framework of FinOL</em>
</p>

## Why should I use FinOL?


1. ``FinOL`` contributes comprehensive datasets spanning diverse market conditions and asset classes to enable large-scale empirical validation;
2. ``FinOL`` contributes the most extensive benchmark results to date for portfolio selection methods, providing the academic community an unbiased performance assessment;
3. ``FinOL`` contributes a user-friendly Python library for data-driven OLPS research, providing a comprehensive toolkit for academics to develop, test, and validate new OLPS methods.

## Installation

### Installing via PIP
``FinOL`` is available on [PyPI](https://pypi.org/project/finol), therefore you can install the latest released version with:
```bash
> pip install finol -t your_own_dir
```

### Installing from source
To install the bleeding edge version, clone this repository with:
```bash
> git clone https://github.com/FinOL/finol
```

## Examples and Tutorials
You can find useful tutorials on how to use ``FinOL`` in the [tutorials](finol/tutorials/README.md) folder.

Here we show a simple application (taken from [tutorial_2](tutorials/tutorial_2.ipynb)): we transform asset "AA" into a 
richer representation.

<p align="center">
<img src="figure/tutorial_2.png" alt>
<em>Visualization of Train Normalization Data for Asset "AA"</em>
</p>


## Using FinOL
To lower the barriers for the research community, ``FinOL`` provides a complete data-training-testing suite with just 
three lines of command.
```python3
from finol.data_layer.data_loader import *
from finol.optimization_layer.model_trainer import *
from finol.evaluation_layer.model_evaluator import *


load_dataset_output = load_dataset()
train_model_output = train_model(load_dataset_output)
evaluate_model_output = evaluate_model(load_dataset_output, train_model_output)
```

## File Structure
```
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
```
## Related Publications


## Contact Us

For inquiries, please get in touch with us at finol.official@gmail.com (Monday to Friday, 9:00 AM to 6:00 PM)


[//]: # (## Useful Links)

 
