.. module:: finol.config

finol.config
============

.. contents::
    :local:


General Configuration
---------------------

.. list-table::
   :header-rows: 1

   * - NAME
     - Description
     - Support Values
   * - ``DEVICE``
     - Specifies the device to be used for computation.
     - ``cuda``, ``cpu``
   * - ``MANUAL_SEED``
     - Sets the seed for random number generation to ensure reproducibility of results.
     - ``Integer``

Data Layer Configuration
------------------------

.. list-table::
   :header-rows: 1

   * - NAME
     - Description
     - Support Values
   * - ``LOAD_LOCAL_DATALOADER``
     - Determines whether to load the dataloader from the local data source.
     - ``Boolean``
   * - ``DATASET_NAME``
     - Specifies the dataset to use (:ref:`supported_datasets`).
     - ``NYSE(O)``, ``NYSE(N)``, ``DJIA``, ``SP500``, ``TSE``, ``SSE``, ``HSI``, ``CMEG``, ``CRYPTO``
   * - ``INCLUDE_OHLCV_FEATURES``
     - Determines whether to include the :ref:`OHLCV_features` in the input data.
     - ``Boolean``
   * - ``INCLUDE_OVERLAP_FEATURES``
     - Determines whether to include the :ref:`overlap_features` in the input data.
     - ``Boolean``
   * - ``INCLUDE_MOMENTUM_FEATURES``
     - Determines whether to include the :ref:`momentum_features` in the input data.
     - ``Boolean``
   * - ``INCLUDE_VOLUME_FEATURES``
     - Determines whether to include the :ref:`volume_features` in the input data.
     - ``Boolean``
   * - ``INCLUDE_CYCLE_FEATURES``
     - Determines whether to include the :ref:`cycle_features` in the input data.
     - ``Boolean``
   * - ``INCLUDE_PRICE_FEATURES``
     - Determines whether to include the :ref:`price_features` in the input data.
     - ``Boolean``
   * - ``INCLUDE_VOLATILITY_FEATURES``
     - Determines whether to include the :ref:`volatility_features` in the input data.
     - ``Boolean``
   * - ``INCLUDE_PATTERN_FEATURES``
     - Determines whether to include the :ref:`pattern_features` in the input data.
     - ``Boolean``
   * - ``INCLUDE_WINDOW_DATA``
     - Determines whether to include the look-back window data in the input data.
     - ``Boolean``
   * - ``WINDOW_SIZE``
     - Specifies the window size use for containing look-back window data.
     - ``Integer``
   * - ``SCALER``
     - Specifies the type of data scaling method to apply to the input data.
     - ``None``, ``StandardScaler``, ``MinMaxScaler``, ``MaxAbsScaler``, ``RobustScaler``, ``WindowStandardScaler``, ``WindowMinMaxScaler``, ``WindowMaxAbsScaler``, ``WindowRobustScaler``
   * - ``BATCH_SIZE``
     - Specifies the batch size to use during model training and validation.
     - ``Integer``

Model Layer Configuration
-------------------------

.. list-table::
   :header-rows: 1

   * - NAME
     - Description
     - Support Values
   * - ``MODEL_NAME``
     - Specifies the type of model to be used. Each model type corresponds to a different neural network architecture.
     - ``CNN``, ``DNN``, ``RNN``, ``LSTM``, ``CNN``, ``Transformer``, ``LSRE-CAAN``, ``AlphaPortfolio``
   * - ``MODEL_PARAMS``
     - A dictionary that contains the model parameters and their values. The keys are parameter names (as strings), and
       the values are either integers or floats.
     - ``Dict[String, Union[Integer, Float]]``
   * - ``MODEL_PARAMS_SPACE``
     -  A tuple of dictionaries, where each dictionary represents a set of model hyperparameters to be explored during
        hyperparameter tuning.
     - ``Tuple[Dict[String, Union[Integer, Float, List]], ...]``


Optimization Layer Configuration
--------------------------------

.. list-table::
   :header-rows: 1

   * - NAME
     - Description
     - Support Values
   * - ``NUM_EPOCHES``
     -
     - ``Integer``
   * - ``SAVE_EVERY``
     -
     - ``Integer``
   * - ``OPTIMIZER_NAME``
     -
     - "Adadelta", "Adagrad", "Adam", "AdamW", "Adamax", "ASGD", "SGD", "RAdam", "Rprop",
                                       "RMSprop", "NAdam", "A2GradExp", "A2GradInc", "A2GradUni", "AccSGD", "AdaBelief",
                                       "AdaBound", "AdaMod", "Adafactor", "AdamP", "AggMo", "Apollo", "DiffGrad", "LARS",
                                       "Lamb", "MADGRAD", "NovoGrad", "PID", "QHAdam", "QHM", "Ranger", "RangerQH", "RangerVA",
                                       "SGDP", "SGDW", "SWATS", "Yogi"
   * - ``LEARNING_RATE``
     -
     - ``Float``
   * - ``CRITERION_NAME``
     -
     - ``"LOG_WEALTH", "LOG_WEALTH_L2_DIVERSIFICATION", "LOG_WEALTH_L2_CONCENTRATION", "L2_DIVERSIFICATION", "L2_CONCENTRATION", "SHARPE_RATIO", "VOLATILITY"``
   * - ``LAMBDA_L2``
     -
     - ``Float``
   * - ``TUNE_PARAMETERS``
     - Determines whether to perform hyperparameter tuning. If set to ``true``, the specified model parameters will be
       tuned during training.
     - ``Boolean``
   * - ``SAMPLER_NAME``
     - Specifies the algorithm to be used for hyperparameter optimization. See `optuna.samplers <https://optuna.readthedocs.io/en/stable/reference/samplers/index.html>`__
       and `"Which sampler should be used?" <https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html#which-sampler-and-pruner-should-be-used>`__ for more details.
     - ``RandomSampler``, ``GridSampler``, ``TPESampler``, ``CmaEsSampler``, ``NSGAIISampler``, ``QMCSampler``, ``GPSampler``, ``BoTorchSampler``, ``BruteForceSampler``
   * - ``NUM_TRIALS``
     - The number of trials to perform during hyperparameter tuning. This determines how many different sets of hyperparameters will be tested.
     - ``Integer``


Evaluation Layer Configuration
------------------------------