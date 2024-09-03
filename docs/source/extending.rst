Extending FinOL
===============

``FinOL`` is designed with extensibility in mind, allowing users to integrate their own models and datasets for benchmarking
purposes. This section provides a step-by-step guide on how to add new methods and datasets to the ``FinOL`` framework.

Adding New Methods
------------------

1. Navigate to ``{ROOT_PATH}\model_layer\CustomModel.py``.
2. Customize your model by extending the ``CustomModel`` class. Ensure it adheres to the interface defined by FinOL for consistency and compatibility.

.. code:: python3

    >>> import torch
    >>> import torch.nn as nn

    >>> from finol.data_layer.scaler_selector import ScalerSelector
    >>> from finol.utils import load_config


    >>> # User-defined model class
    >>> class CustomModel(nn.Module):
    >>>     """
    >>>     CustomModel as a base neural network model for portfolio selection.

    >>>     Users can extend this class and implement their desired model architecture and functionality.

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
    >>>         # x = x.view(batch_size, num_assets, window_size, num_features_original)

    >>>         """Input Transformation"""
    >>>         if self.config["SCALER"].startswith("Window"):
    >>>             x = ScalerSelector().window_normalize(x)

    >>>         ...

    >>>         final_scores = x

    >>>         return final_scores


3. Define the necessary hyper-parameters in ``{ROOT_PATH}\config.json`` at ``config["MODEL_PARAMS"]["CustomModel"]``.

.. code:: json

    "MODEL_PARAMS": {
        // other models...
        "CustomModel": {
            "PARAMETER1": 4,
            "PARAMETER2": 128,
            // other hyper-parameters...
        }
    },


4. (Optional) If you want ``FinOL`` to automatically tune the hyper-parameters of your custom model,
specify the range of different parameters in the ``MODEL_PARAMS_SPACE["CustomModel"]`` section of the ``config.json`` file.

.. code:: json

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

Please refer to the example implementation in :mod:`~finol.data_layer.CustomModel` for guidance on the expected structure and
interface of your custom model class. Additionally, the FinOL documentation provides detailed tutorials and
API references to help you get started.