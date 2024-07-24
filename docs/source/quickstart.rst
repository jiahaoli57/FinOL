Quickstart
==========

.. toctree::
   :maxdepth: 1

This guide will help you get started with ``FinOL``.
To lower the barriers for the research community,
``FinOL`` provides a complete data-training-testing suite
with just three lines of command.

Command Line Usage
------------------

.. code:: python3

   from finol.data_layer.DatasetLoader import DatasetLoader
   from finol.optimization_layer.ModelTrainer import ModelTrainer
   from finol.evaluation_layer.ModelEvaluator import ModelEvaluator


   load_dataset_output = DatasetLoader().load_dataset()
   train_model_output = ModelTrainer(load_dataset_output).train_model()
   evaluate_model_output = ModelEvaluator(load_dataset_output, train_model_output).evaluate_model()

Before running the above commands, users can first
configure some parameters through the config file to customize
the usage according to their needs. For example, setting the
device to CPU, selecting a different dataset, adjusting the
data preprocessing parameters, and choosing a different model, etc.
The specific configuration method is as follows:

.. code:: python3

   from finol.utils import load_config, update_config

   config = load_config()
   config["DEVICE"] = "cpu"
   config["DATASET_NAME"] = "DJIA"
   config["DATA_AUGMENTATION_CONFIG"]["WINDOW_DATA"]["WINDOW_SIZE"] = 15
   config["SCALER"] = "WindowMinMaxScaler"
   config["MODEL_NAME"] = "DNN"
   config["TUNE_PARAMETERS"] = True
   update_config(config)

|Open in Colab|

GUI Usage
---------

In addition to the command line usage, ``FinOL`` also providesp a GUI
interface that allows users to achieve the same functionality
as the command line usage in a more intuitive and visual way.
The GUI interace includes options for dataset selection,
model configuration, training, and evaluation,
allowing users to easily customize the parameters and run the
exeriments without the need to write any code.

.. image:: ../../images/finol_gui.png
    :align: center
.. centered:: *Overall Framework of FinOL GUI*

Data Layer panel
~~~~~~~~~~~~~~~~

In the data layer panel, users can choose from a variety of available datasets,
such as DJIA, S&P500, and Nasdaq Composite.
.. The GUI also displays information about each dataset,
such as the number of samples and features, to help users make an informed decision.

Model Layer panel
~~~~~~~~~~~~~~~~~

The model layer panel allows users to select the model architecture and other settings.
Users can choose from a range of pre-defined models or create their own custom models.
.. The xx provides detailed information about each model and its parameters,
making it easier for users to understand and customize the model.

Optimization Layer panel
~~~~~~~~~~~~~~~~~~~~~~~~

The optimization layer panel enables users to monitor the training process.
Users can view real-time metrics, such as loss,
and adjust the training parameters as needed.

Evaluation Layer panel
~~~~~~~~~~~~~~~~~~~~~~

The evaluation layer panel enables users to evaluate the trained model's performance.
The GUI also provides visualizations of the model's performance,
making it easier to interpret the results.

By using the ``FinOL`` GUI, users can quickly and easily configure, train, and
evaluate financial models without the need to write complex code.  The intuitive interface and visual feedback make the
process more accessible and user-friendly, especially for researchers and
practitioners who are new to the field of financial modeling.


.. |Open in Colab| image:: https://colab.research.google.com/assets/colab-badge.svg
  :target: https://colab.research.google.com/github/jiahaoli57/FinOL/blob/main/finol/tutorials/tutorial_quickstart.ipynb

