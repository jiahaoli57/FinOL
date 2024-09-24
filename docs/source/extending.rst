Extend FinOL
============

``FinOL`` is designed with extensibility in mind, allowing users to integrate their own models and datasets for benchmarking
purposes. This section provides a step-by-step guide on how to extend the ``FinOL`` framework.

.. contents::
    :local:

Adding New Dataset
------------------

To integrate your own datasets into ``FinOL``, follow these steps:

1. Navigate to the ``{ROOT_PATH}\data\datasets\CustomDataset`` directory.
2. Create .xlsx files for different assets in the following format:

.. code-block:: python

     +------------+----------+----------+----------+----------+---------+
     | DATE       | OPEN     |  HIGH    | LOW      | CLOSE    | VOLUME  |
     |------------+----------+----------+----------+----------+---------|
     | 2017-11-09 | 0.025160 | 0.035060 | 0.025006 | 0.032053 | 1871620 |
     | 2017-11-10 | 0.032219 | 0.033348 | 0.026450 | 0.027119 | 6766780 |
     | 2017-11-11 | 0.026891 | 0.029658 | 0.025684 | 0.027437 | 5532220 |
     | 2017-11-12 | 0.027480 | 0.027952 | 0.022591 | 0.023977 | 7280250 |
     | 2017-11-13 | 0.024364 | 0.026300 | 0.023495 | 0.025807 | 4419440 |
     | 2017-11-14 | 0.025797 | 0.026788 | 0.025342 | 0.026230 | 3033290 |
     | 2017-11-15 | 0.026116 | 0.027773 | 0.025261 | 0.026445 | 6858800 |
     | ......     | ......   | ......   | ......   | ......   | ......  |
     | 2024-02-29 | 0.630859 | 0.705280 | 0.625720 | 0.655646 | 1639531 |
     | 2024-03-01 | 0.655440 | 0.719080 | 0.654592 | 0.719080 | 9353798 |
     +-----------+-----------+----------+----------+----------+---------+

3. For each asset, ensure that the data are correctly formatted and there are no missing values.

4. Define the configuration for your custom dataset in the ``{ROOT_PATH}\config.json`` file, under the ``config["DATASET_SPLIT_CONFIG"]["CustomModel"]``, ``config["BATCH_SIZE"]["CustomModel"]``, and ``config["NUM_DAYS_PER_YEAR"]["CustomModel"]`` sections. For splitting your dataset, it is recommended to use a ratio of 0.6:0.2:0.2 for training, validation, and testing datasets, respectively.


.. code-block:: json
    :caption: config.json

    "DATASET_SPLIT_CONFIG": {
        // other datasets...
        "CustomDataset": {
            "TRAIN_START_TIMESTAMP": "",
            "TRAIN_END_TIMESTAMP": "",
            "VAL_START_TIMESTAMP": "",
            "VAL_END_TIMESTAMP": "",
            "TEST_START_TIMESTAMP": "",
            "TEST_END_TIMESTAMP": ""
        }
    },
    "BATCH_SIZE": {
        // other datasets...
        "CustomDataset": 64  // set an appropriate batch size
    },
    "NUM_DAYS_PER_YEAR": {
        // other datasets...
        "CustomDataset": 252  // set an appropriate number of days per year
    }


.. note::
    Instead of customizing the dataset yourself, we recommend that you raise an issue or contact us by email so we can
    evaluate and potentially include your dataset in the ``FinOL`` project.
    This ensures the baseline results are supported.

Adding New Method
-----------------

1. Navigate to the ``{ROOT_PATH}\model_layer\CustomModel.py`` file in the ``FinOL`` codebase.
2. Customize your model by extending the :mod:`~finol.model_layer.CustomModel` class. This is where you will implement the logic for your custom data-driven OLPS model. Ensure it adheres to the interface defined by ``FinOL`` for consistency and compatibility.

.. code-block:: python3
    :caption: CustomModel.py

    >>> import torch
    >>> import torch.nn as nn

    >>> from einops import rearrange
    >>> from finol.data_layer.scaler_selector import ScalerSelector
    >>> from finol.utils import load_config


    >>> # User-defined model class
    >>> class CustomModel(nn.Module):
    >>>     """
    >>>     Class to serve as a base neural network model for portfolio selection. This class provides users with a framework
    >>>     to extend and implement their own model architectures and functionality,
    >>>     allowing for customization to meet specific requirements and objectives in financial modeling.

    >>>     :param model_args: Dictionary containing model arguments, such as the number of features.
    >>>     :param model_params: Dictionary containing model hyper-parameters, such as the parameter1, parameter2, etc.

    >>>     Example:
    >>>         .. code:: python
    >>>         >>> from finol.data_layer.dataset_loader import DatasetLoader
    >>>         >>> from finol.model_layer.model_instantiator import ModelInstantiator
    >>>         >>> from finol.utils import load_config, update_config, portfolio_selection
    >>>         >>>
    >>>         >>> # Configuration
    >>>         >>> config = load_config()
    >>>         >>> config["MODEL_NAME"] = "CustomModel"
    >>>         >>> config["MODEL_PARAMS"]["CustomModel"]["PARAMETER1"] = 2
    >>>         >>> config["MODEL_PARAMS"]["CustomModel"]["PARAMETER1"] = 128
    >>>         >>> update_config(config)
    >>>         >>>
    >>>         >>> # Data Layer
    >>>         >>> load_dataset_output = DatasetLoader().load_dataset()
    >>>         >>>
    >>>         >>> # Model Layer & Optimization Layer
    >>>         >>> ...
    >>>         >>> model = ModelInstantiator(load_dataset_output).instantiate_model()
    >>>         >>> print(f"model: {model}")
    >>>         >>> ...
    >>>         >>> train_loader = load_dataset_output["train_loader"]
    >>>         >>> for i, data in enumerate(train_loader, 1):
    >>>         ...     x_data, label = data
    >>>         ...     final_scores = model(x_data.float())
    >>>         ...     portfolio = portfolio_selection(final_scores)
    >>>         ...     print(f"batch {i} input shape: {x_data.shape}")
    >>>         ...     print(f"batch {i} label shape: {label.shape}")
    >>>         ...     print(f"batch {i} output shape: {portfolio.shape}")
    >>>         ...     print("-"*50)

    >>>     .. warning::
    >>>         When users define their own model, besides modifying this class, they must add different parameter keys and values
    >>>         in the ``config.json`` at the location ``config["MODEL_PARAMS"]["CustomModel"]``. Similarly, if users want to implement
    >>>         automatic hyper-parameters tuning for their custom model, they also need to specify the range and type of different
    >>>         parameters at ``config["MODEL_PARAMS_SPACE"]["CustomModel"]``
    >>>     """

    >>>     def __init__(self, model_args, model_params):
    >>>         super().__init__()
    >>>         self.config = load_config()
    >>>         self.model_args = model_args
    >>>         self.model_parms = model_params
    >>>         # Define your model architecture here

    >>>     def forward(self, x: torch.Tensor) -> torch.Tensor:
    >>>         """
    >>>         Forward pass of the model.

    >>>         :param x: Input tensor of shape `(batch_size, num_assets, num_features_augmented)`.
    >>>         :return: Output tensor of shape `(batch_size, num_assets)` containing the predicted scores for each asset.
    >>>         """
    >>>         batch_size, num_assets, num_features_augmented = x.shape

    >>>         """Input Transformation"""
    >>>         x = x.view(batch_size, num_assets, self.model_args["window_size"], self.model_args["num_features_original"])
    >>>         x = rearrange(x, "b m n d -> (b m) n d")
    >>>         """Input Transformation"""
    >>>         if self.config["SCALER"].startswith("Window"):
    >>>             x = ScalerSelector().window_normalize(x)

    >>>         ...

    >>>         final_scores = x

    >>>         return final_scores


3. Define the necessary hyper-parameters in ``{ROOT_PATH}\config.json`` at ``config["MODEL_PARAMS"]["CustomModel"]``.

.. code-block:: json
    :caption: config.json

    "MODEL_PARAMS": {
        // other models...
        "CustomModel": {
            "PARAMETER1": 4,
            "PARAMETER2": 128,
            // other hyper-parameters...
        }
    },


4. (Optional) If you want ``FinOL`` to automatically tune the hyper-parameters of your custom model, specify the range of different parameters in the ``MODEL_PARAMS_SPACE["CustomModel"]`` section of the ``config.json`` file.

.. code-block:: json
    :caption: config.json

    "MODEL_PARAMS_SPACE": {
        // other models...
        "CustomModel": {
            "PARAMETER1": {
                "type": "int",
                "range": [
                    1,
                    4
                ],
                "step": 1
            },
            "PARAMETER2": {
                "type": "int",
                "range": [
                    32,
                    256
                ],
                "step": 32
            },
            // other hyper-parameters...
        }
    }

Please refer to the example implementation in :mod:`~finol.model_layer.CustomModel` for guidance on the expected structure and
interface of your custom model class. Additionally, the ``FinOL`` documentation provides detailed tutorials and
API references to help you get started.

Adding New Criterion
--------------------

1. Navigate to the ``{ROOT_PATH}\optimization_layer\criterion_selector.py`` file in the ``FinOL`` codebase.
2. Locate the :mod:`~finol.optimization_layer.CriterionSelector` class and define your own custom investment criterion by rewriting the :func:`~finol.optimization_layer.CriterionSelector.compute_custom_criterion_loss` method. Ensure it adheres to the interface defined by ``FinOL`` for consistency and compatibility.

.. code-block:: python3
    :caption: criterion_selector.py

    >>> import time
    >>> import torch

    >>> from finol.utils import load_config


    >>> class CriterionSelector:
    >>>     """
    >>>     Class to select and compute different loss criteria for portfolio selection.
    >>>     """
    >>>     def __init__(self) -> None:
    >>>         self.config = load_config()
    >>>         self.criterion_dict = {
    >>>             "LogWealth": self.compute_log_wealth_loss,
    >>>             "LogWealthL2Diversification": self.compute_log_wealth_l2_diversification_loss,
    >>>             "LogWealthL2Concentration": self.compute_log_wealth_l2_concentration_loss,
    >>>             "L2Diversification": self.compute_l2_diversification_loss,
    >>>             "L2Concentration": self.compute_l2_concentration_loss,
    >>>             "SharpeRatio": self.compute_sharpe_ratio_loss,
    >>>             "Volatility": self.compute_volatility_loss,
    >>>             "CustomCriterion": self.compute_custom_criterion_loss,
    >>>         }

    >>>         ...

    >>>     def compute_custom_criterion_loss(self, portfolios: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    >>>         """
    >>>         Compute the ``CustomCriterion`` loss,  which is left for the user to define.

    >>>         This loss function is a placeholder for the user to implement their own custom loss criterion.

    >>>         :param portfolios: Portfolio weights tensor of shape (batch_size, num_assets).
    >>>         :param labels: Label tensor representing asset returns of shape (batch_size, num_assets).
    >>>         :return: ``CustomCriteria`` loss tensor, representing the user-defined loss criterion.
    >>>         """
    >>>         # This is a placeholder for the user to implement their own custom loss function.
    >>>         # The implementation of the custom loss function is left to the user.
    >>>         loss = torch.tensor(0.0, requires_grad=True)
    >>>         return loss

    >>>     def __call__(self, portfolios: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    >>>         criterion_cls = self.criterion_dict.get(self.config["CRITERION_NAME"], None)
    >>>         if criterion_cls is None:
    >>>            raise ValueError(f"Invalid criterion name: {self.config['CRITERION_NAME']}. Supported criteria are: {self.criterion_dict.keys()}")
    >>>         return criterion_cls(portfolios, labels)