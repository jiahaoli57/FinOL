.. module:: finol.config

finol.config
============


Data Layer Configuration
------------------------

``DEVICE``
~~~~~~~~~~

:Type: str
:Options: ``auto``, ``cuda``, ``cpu``.
:Description: Specifies the device to be used for computation.

``MANUAL_SEED``
~~~~~~~~~~~~~~~

:Type: int
:Options: Any non-negative integer.
:Description: Sets the seed for random number generation to ensure reproducibility of results.

``LOAD_LOCAL_DATALOADER``
~~~~~~~~~~~~~~~~~~~~~~~~~

:Type: bool
:Options: ``True``, ``False``.
:Description: Determines whether to load the dataloader from the local data source.

``DATASET_NAME``
~~~~~~~~~~~~~~~~

:Type: str
:Options: ``NYSE(O)``, ``NYSE(N)``, ``DJIA``, ``SP500``, ``TSE``, ``SSE``, ``HSI``, ``CMEG``, ``CRYPTO``.
:Description: Specifies the dataset to use (:ref:`supported_datasets`).

``INCLUDE_OHLCV_FEATURES``
~~~~~~~~~~~~~~~~~~~~~~~~~~

:Type: bool
:Options: ``True``, ``False``.
:Description: Determines whether to include the :ref:`OHLCV_features` in the input data.

``INCLUDE_OVERLAP_FEATURES``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:Type: bool
:Options: ``True``, ``False``.
:Description: Determines whether to include the :ref:`overlap_features` in the input data.

``INCLUDE_MOMENTUM_FEATURES``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:Type: bool
:Options: ``True``, ``False``.
:Description: Determines whether to include the :ref:`momentum_features` in the input data.

``INCLUDE_VOLUME_FEATURES``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

:Type: bool
:Options: ``True``, ``False``.
:Description: Determines whether to include the :ref:`volume_features` in the input data.

``INCLUDE_CYCLE_FEATURES``
~~~~~~~~~~~~~~~~~~~~~~~~~~

:Type: bool
:Options: ``True``, ``False``.
:Description: Determines whether to include the :ref:`cycle_features` in the input data.

``INCLUDE_PRICE_FEATURES``
~~~~~~~~~~~~~~~~~~~~~~~~~~

:Type: bool
:Options: ``True``, ``False``.
:Description: Determines whether to include the :ref:`price_features` in the input data.

``INCLUDE_VOLATILITY_FEATURES``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:Type: bool
:Options: ``True``, ``False``.
:Description: Determines whether to include the :ref:`volatility_features` in the input data.

``INCLUDE_PATTERN_FEATURES``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:Type: bool
:Options: ``True``, ``False``.
:Description: Determines whether to include the :ref:`pattern_features` in the input data.

``INCLUDE_WINDOW_DATA``
~~~~~~~~~~~~~~~~~~~~~~~

:Type: bool
:Options: ``True``, ``False``.
:Description: Determines whether to include the look-back window data in the input data.

``WINDOW_SIZE``
~~~~~~~~~~~~~~~

:Type: int
:Options: Any positive integer.
:Description: Specifies the window size use for containing look-back window data.

``SCALER``
~~~~~~~~~~

:Type: str
:Options: ``None``, ``StandardScaler``, ``MinMaxScaler``, ``MaxAbsScaler``, ``RobustScaler``, ``WindowStandardScaler``, ``WindowMinMaxScaler``, ``WindowMaxAbsScaler``, ``WindowRobustScaler``.
:Description: Specifies the type of data scaling method to apply to the input data.

``BATCH_SIZE``
~~~~~~~~~~~~~~

:Type: int
:Options: Any positive integer.
:Description: Specifies the batch size to use during model training and validation.


Model Layer Configuration
-------------------------

``MODEL_NAME``
~~~~~~~~~~~~~~

:Type: str
:Options: ``CNN``, ``DNN``, ``RNN``, ``LSTM``, ``CNN``, ``Transformer``, ``LSRE-CAAN``, ``AlphaPortfolio``.
:Description: Specifies the type of model to be used. Each model corresponds to a different neural network architecture.

``MODEL_PARAMS``
~~~~~~~~~~~~~~~~

:Type: dict
:Options: The keys in the dictionary correspond to the names of the model parameters, and the values correspond to the desired parameter values.
:Description: Specifies the model parameters and their values.

``MODEL_PARAMS_SPACE``
~~~~~~~~~~~~~~~~~~~~~~

:Type: dict
:Options: The keys in the dictionary correspond to the names of the model parameters, and the values correspond to the range of the parameter values.
:Description: Specifies the set of model hyper-parameters to be explored during hyper-parameters tuning.

Optimization Layer Configuration
--------------------------------

``NUM_EPOCHES``
~~~~~~~~~~~~~~~

:Type: int
:Options:  Any positive integer.
:Description: Specifies the number of training epochs to run.

``SAVE_EVERY``
~~~~~~~~~~~~~~

:Type: int
:Options: Any positive integer.
:Description: Specifies the number of epochs after which to save the model checkpoint.

``OPTIMIZER_NAME``
~~~~~~~~~~~~~~~~~~

:Type: str
:Options: ``Adadelta``, ``Adagrad``, ``Adam``, ``AdamW``, ``Adamax``, ``ASGD``, ``SGD``, ``RAdam``, ``Rprop``, ``RMSprop``, ``NAdam``, ``A2GradExp``, ``A2GradInc``, ``A2GradUni``, ``AccSGD``, ``AdaBelief``, ``AdaBound``, ``AdaMod``, ``Adafactor``, ``AdamP``, ``AggMo``, ``Apollo``, ``DiffGrad``, ``LARS``, ``Lamb``, ``MADGRAD``, ``NovoGrad``, ``PID``, ``QHAdam``, ``QHM``, ``Ranger``, ``RangerQH``, ``RangerVA``, ``SGDP``, ``SGDW``, ``SWATS``, ``Yogi``.
:Description: Specifies the optimizer to use during training.

``LEARNING_RATE``
~~~~~~~~~~~~~~~~~

:Type: float
:Options: Any positive float.
:Description: Specifies the step size at each iteration while moving toward a minimum/maximum of a criterion.

``CRITERION_NAME``
~~~~~~~~~~~~~~~~~~

:Type: str
:Options: ``LogWealth``, ``LogWealthL2Diversification``, ``LogWealthL2Concentration``, ``L2Diversification``, ``L2Concentration``, ``SharpeRatio``, ``Volatility``.
:Description: Specifies the name of the criterion to be used during training.

``LAMBDA_L2``
~~~~~~~~~~~~~

:Type: float
:Options: Any non-negative float.
:Description: Specifies the strength of the L2 regularization. Required only when the ``CRITERION_NAME`` is set to ``LogWealthL2Diversification`` or ``LogWealthL2Concentration``.

``TUNE_PARAMETERS``
~~~~~~~~~~~~~~~~~~~

:Type: bool
:Options: ``Ture``, ``False``.
:Description: Determines whether to perform hyper-parameters tuning.

``NUM_TRIALS``
~~~~~~~~~~~~~~

:Type: int
:Options: Any positive integer.
:Description: Specifies the number of trials to perform during hyper-parameters tuning. This determines how many different sets of hyper-parameters will be tested.

``SAMPLER_NAME``
~~~~~~~~~~~~~~~~

:Type: str
:Options:  ``BruteForceSampler``, ``CmaEsSampler``, ``GridSampler``, ``NSGAIISampler``, ``NSGAIIISampler``, ``QMCSampler``, ``RandomSampler``, ``TPESampler``, ``GPSampler``.
:Description: Specifies the algorithm to be used for hyper-parameters tuning. See `optuna.samplers <https://optuna.readthedocs.io/en/stable/reference/samplers/index.html>`__ and `Which sampler should be used? <https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html#which-sampler-and-pruner-should-be-used>`__ for more details.

``PRUNER_NAME``
~~~~~~~~~~~~~~~

:Type: str
:Options:  ``HyperbandPruner``, ``MedianPruner``,  ``NopPruner``, ``PatientPruner``, ``SuccessiveHalvingPruner``, ``WilcoxonPruner``.
:Description: Specifies the pruner to be used for hyper-parameters tuning. See `optuna.pruners <https://optuna.readthedocs.io/en/stable/reference/pruners.html>`__ for more details.

``WRAPPED_PRUNER_NAME``
~~~~~~~~~~~~~~~~~~~~~~~

:Type: str
:Options:  ``HyperbandPruner``, ``MedianPruner``,  ``SuccessiveHalvingPruner``, ``WilcoxonPruner``.
:Description: Specifies the wrapped pruner to be used for hyper-parameters tuning. Required only when the ``PRUNER_NAME`` is set to ``PatientPruner``.

Evaluation Layer Configuration
------------------------------

``PLOT_LANGUAGE``
~~~~~~~~~~~~~~~~~

:Type: str
:Options: ``en`` (English), ``zh_CN`` (Chinese Simple), ``zh_TW`` (Chinese Traditional).
:Description: Specifies the language to use for plot labels and legends.

``PROP_WINNERS``
~~~~~~~~~~~~~~~~

:Type: float
:Options: A value between 0 and 1.
:Description: Specifies the proportion of winner assets to be invested during the actual investment process. This parameter determines how many of the best-performing assets will be invested.

``INCLUDE_INTERPRETABILITY_ANALYSIS``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:Type: bool
:Options: ``True``, ``False``.
:Description: Determines whether to include an interpretability analysis as part of the overall analysis. The interpretability analysis aims to provide insights into the features that drive the generation of the portfolios.

``INCLUDE_ECONOMIC_DISTILLATION``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:Type: bool
:Options: ``True``, ``False``.
:Description: Determines whether to include an economic distillation analysis as part of the overall analysis. The economic distillation analysis aims to identify the most important economic features that influence portfolio performance, allowing for a more focused and interpretable model.

``PROP_DISTILLED_FEATURES``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

:Type: float
:Options: A value between 0 and 1.
:Description: Specifies the proportion of the most important features to be retained after the economic distillation process. This parameter determines how many of the original features will be used in the economic distillation model, with the goal of creating a more interpretable and efficient model.

``DISTILLER_NAME``
~~~~~~~~~~~~~~~~~~

:Type: str
:Options: ``LinearRegression``, ``Ridge``, ``RidgeCV``, ``SGDRegressor``, ``ElasticNet``, ``ElasticNetCV``, ``Lars``, ``LarsCV``, ``Lasso``, ``LassoCV``, ``LassoLars``, ``LassoLarsCV``, ``LassoLarsIC``, ``OrthogonalMatchingPursuit``, ``OrthogonalMatchingPursuitCV``, ``ARDRegression``, ``BayesianRidge``, ``HuberRegressor``, ``QuantileRegressor``, ``RANSACRegressor``, ``TheilSenRegressor``, ``PoissonRegressor``, ``TweedieRegressor``, ``GammaRegressor``, ``PassiveAggressiveRegressor``.
:Description: Specifies the feature distiller to be used in the economic distillation analysis. This parameter determines the specific method that will be used to identify the most important features from the original set of input variables.

``Y_NAME``
~~~~~~~~~~

:Type: str
:Options: ``Scores``, ``Portfolios``.
:Description: Specifies the target variable for the economic distillation model.