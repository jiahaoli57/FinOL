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

| Update                     | Status                                                                                      |
|----------------------------|---------------------------------------------------------------------------------------------|
| Release  ``FinOL``  v0.0.1 | [Released v0.0.1](https://github.com/jiahaoli57/finol/releases/tag/v0.0.1) on 17 March 2024 |

</div>


## Outline

- [FinOL: Towards Open Benchmarking for Data-Driven Online Portfolio Selection](#finol)
  - [Outline](#outline)
  - [About](#about)
  - [Why should I use FinOL?](#why-should-i-use-finOL?)
  - [Installation](#installation)
  - [Examples and Tutorials](#examples-and-tutorials)
  - [Using FinOL](#using-FinOL)
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

## Contact Us

For inquiries, please get in touch with us at finol.official@gmail.com (Monday to Friday, 9:00 AM to 6:00 PM)


[//]: # (## Useful Links)

 
