import time
import json
import tkinter as tk
from tkinter import messagebox, ttk
from finol.data_layer import DatasetLoader
from finol.optimization_layer import ModelTrainer
from finol.evaluation_layer import ModelEvaluator
from finol.utils import load_config, update_config


# def format_string(input_string):
#     words = input_string.split("_")
#     formatted_words = [word.capitalize() for word in words]
#     formatted_string = " ".join(formatted_words)
#
#     return formatted_string


class FinOLApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("FinOL: Towards Open Benchmarking for Data-Driven Online Portfolio Selection")
        self.create_widgets()

        self.experiment_details = {}

    def run(self):
        self.root.mainloop()

    def create_widgets(self):
        ###############################
        # General Layer Configuration #
        ###############################
        self.general_config_frame = tk.LabelFrame(self.root, text="General Configuration", font=("Helvetica", 10, "bold"))
        self.general_config_frame.pack(padx=10, pady=1, fill="none")

        # DEVICE
        self.DEVICE_options = ["cpu", "cuda"]
        tk.Label(self.general_config_frame, text="Select Device:").grid(row=0, column=0, padx=10, pady=1)
        self.DEVICE_var = tk.StringVar()
        self.DEVICE_dropdown = ttk.Combobox(self.general_config_frame, textvariable=self.DEVICE_var, values=self.DEVICE_options)
        self.DEVICE_dropdown.grid(row=0, column=1, padx=10, pady=1)
        self.DEVICE_dropdown.current(0)

        # MANUAL_SEED
        tk.Label(self.general_config_frame, text="Set Seed:").grid(row=0, column=2, padx=10, pady=1)
        self.MANUAL_SEED_var = tk.IntVar()
        self.MANUAL_SEED_entry = tk.Entry(self.general_config_frame, textvariable=self.MANUAL_SEED_var)
        self.MANUAL_SEED_entry.grid(row=0, column=3, padx=10, pady=1)
        self.MANUAL_SEED_var.set(0)

        ############################
        # Data Layer Configuration #
        ############################
        self.data_config_frame = tk.LabelFrame(self.root, text="Data Layer Configuration", font=("Helvetica", 10, "bold"))
        self.data_config_frame.pack(padx=10, pady=1, fill="none")

        # DATASET_NAME
        self.DATASET_NAME_options = ["NYSE(O)", "NYSE(N)", "DJIA", "SP500", "TSE", "SSE", "HSI", "CMEG", "CRYPTO"]
        tk.Label(self.data_config_frame, text="Select Dataset:").grid(row=0, column=0, padx=10, pady=1)
        self.DATASET_NAME_var = tk.StringVar()
        self.DATASET_NAME_dropdown = ttk.Combobox(self.data_config_frame, textvariable=self.DATASET_NAME_var, values=self.DATASET_NAME_options)
        self.DATASET_NAME_dropdown.grid(row=0, column=1, padx=10, pady=1)
        self.DATASET_NAME_dropdown.current(0)

        # SCALER
        self.SCALER_options = ["None", "StandardScaler", "MinMaxScaler", "MaxAbsScaler", "RobustScaler",
                       "WindowStandardScaler", "WindowMinMaxScaler", "WindowMaxAbsScaler", "WindowRobustScaler"]
        tk.Label(self.data_config_frame, text="Select Scaler:").grid(row=0, column=2, padx=10, pady=1)
        self.SCALER_var = tk.StringVar()
        self.SCALER_dropdown = ttk.Combobox(self.data_config_frame, textvariable=self.SCALER_var, values=self.SCALER_options)
        self.SCALER_dropdown.grid(row=0, column=3, padx=10, pady=1)
        self.SCALER_dropdown.current(5)

        separator = ttk.Separator(self.data_config_frame,  orient="horizontal")
        separator.grid(row=1, column=0, columnspan=4, padx=10, pady=1, sticky="ew")
        separator_label = ttk.Label(self.data_config_frame, text="Auto Feature Engineering")
        separator_label.grid(row=1, column=0, columnspan=4, padx=10, pady=1)

        # INCLUDE_OHLCV_FEATURES
        self.INCLUDE_OHLCV_FEATURES_var = tk.BooleanVar(value=True)
        self.INCLUDE_OHLCV_FEATURES_chk = tk.Checkbutton(self.data_config_frame, text="Include OHLCV Features", variable=self.INCLUDE_OHLCV_FEATURES_var)
        self.INCLUDE_OHLCV_FEATURES_chk.grid(row=2, column=0)

        # INCLUDE_OVERLAP_FEATURES
        self.INCLUDE_OVERLAP_FEATURES_var = tk.BooleanVar(value=True)
        self.INCLUDE_OVERLAP_FEATURES_chk = tk.Checkbutton(self.data_config_frame, text="Include Overlap Features", variable=self.INCLUDE_OVERLAP_FEATURES_var)
        self.INCLUDE_OVERLAP_FEATURES_chk.grid(row=2, column=1)

        # INCLUDE_MOMENTUM_FEATURES
        self.INCLUDE_MOMENTUM_FEATURES_var = tk.BooleanVar(value=True)
        self.INCLUDE_MOMENTUM_FEATURES_chk = tk.Checkbutton(self.data_config_frame, text="Include Momentum Features", variable=self.INCLUDE_MOMENTUM_FEATURES_var)
        self.INCLUDE_MOMENTUM_FEATURES_chk.grid(row=2, column=2)

        # INCLUDE_VOLUME_FEATURES
        self.INCLUDE_VOLUME_FEATURES_var = tk.BooleanVar(value=True)
        self.INCLUDE_VOLUME_FEATURES_chk = tk.Checkbutton(self.data_config_frame, text="Include Volume Features", variable=self.INCLUDE_VOLUME_FEATURES_var)
        self.INCLUDE_VOLUME_FEATURES_chk.grid(row=2, column=3)

        # INCLUDE_CYCLE_FEATURES
        self.INCLUDE_CYCLE_FEATURES_var = tk.BooleanVar(value=True)
        self.INCLUDE_CYCLE_FEATURES_chk = tk.Checkbutton(self.data_config_frame, text="Include Cycle Features", variable=self.INCLUDE_CYCLE_FEATURES_var)
        self.INCLUDE_CYCLE_FEATURES_chk.grid(row=3, column=0)

        #  INCLUDE_PRICE_FEATURES
        self.INCLUDE_PRICE_FEATURES_var = tk.BooleanVar(value=True)
        self.INCLUDE_PRICE_FEATURES_chk = tk.Checkbutton(self.data_config_frame, text="Include Price Features", variable=self.INCLUDE_PRICE_FEATURES_var)
        self.INCLUDE_PRICE_FEATURES_chk.grid(row=3, column=1)

        # INCLUDE_VOLATILITY_FEATURES
        self.INCLUDE_VOLATILITY_FEATURES_var = tk.BooleanVar(value=True)
        self.INCLUDE_VOLATILITY_FEATURES_chk = tk.Checkbutton(self.data_config_frame, text="Include Volatility Features", variable=self.INCLUDE_VOLATILITY_FEATURES_var)
        self.INCLUDE_VOLATILITY_FEATURES_chk.grid(row=3, column=2)

        #  INCLUDE_PATTERN_FEATURES
        self.INCLUDE_PATTERN_FEATURES_var = tk.BooleanVar(value=True)
        self.INCLUDE_PATTERN_FEATURES_chk = tk.Checkbutton(self.data_config_frame, text="Include Pattern Features", variable=self.INCLUDE_PATTERN_FEATURES_var)
        self.INCLUDE_PATTERN_FEATURES_chk.grid(row=3, column=3)

        separator = ttk.Separator(self.data_config_frame,  orient="horizontal")
        separator.grid(row=4, column=0, columnspan=4, padx=10, pady=1, sticky="ew")
        separator_label = ttk.Label(self.data_config_frame, text="Data Augmentation")
        separator_label.grid(row=4, column=0, columnspan=4, padx=10, pady=1)

        # INCLUDE_WINDOW_DATA
        self.INCLUDE_WINDOW_DATA_var = tk.BooleanVar(value=True)
        self.INCLUDE_WINDOW_DATA_chk = tk.Checkbutton(self.data_config_frame, text="Include Window Data", variable=self.INCLUDE_WINDOW_DATA_var, command=self.base_on_INCLUDE_WINDOW_DATA)
        self.INCLUDE_WINDOW_DATA_chk.grid(row=5, column=0)

        # WINDOW_SIZE
        tk.Label(self.data_config_frame, text="Set Window Size:").grid(row=5, column=1, padx=10, pady=1)
        self.WINDOW_SIZE_var = tk.IntVar()
        self.WINDOW_SIZE_entry = tk.Entry(self.data_config_frame, textvariable=self.WINDOW_SIZE_var)
        self.WINDOW_SIZE_entry.grid(row=5, column=2)
        self.WINDOW_SIZE_var.set(10)

        separator = ttk.Separator(self.data_config_frame,  orient="horizontal")
        separator.grid(row=6, column=0, columnspan=4, padx=10, pady=1, sticky="ew")

        # LOAD_LOCAL_DATALOADER
        self.LOAD_LOCAL_DATALOADER_var = tk.BooleanVar(value=False)
        self.LOAD_LOCAL_DATALOADER_checkbox = tk.Checkbutton(self.data_config_frame, text="Load Local Dataloader", variable=self.LOAD_LOCAL_DATALOADER_var, command=self.base_on_LOAD_LOCAL_DATALOADER)
        self.LOAD_LOCAL_DATALOADER_checkbox.grid(row=7, column=0, padx=10, pady=1)

        # load_button
        self.load_button = tk.Button(self.root, text="Load Dataset", command=self.load_dataset)
        self.load_button.pack(padx=10, pady=3)

        #############################
        # Model Layer Configuration #
        #############################
        self.model_config_frame = tk.LabelFrame(self.root, text="Model Layer Configuration", font=("Helvetica", 10, "bold"))
        self.model_config_frame.pack(padx=10, pady=1, fill="none")

        # ttk.Label(self.model_config_frame, text=" "*150).grid(row=0, column=0, columnspan=4, padx=10, pady=0)
        # ttk.Label(self.model_config_frame, text=" "*150).grid(row=100, column=0, columnspan=4, padx=10, pady=0)

        # MODEL_NAME
        self.MODEL_NAME_options = ["AlphaPortfolio", "CNN", "DNN", "LSRE-CAAN", "LSTM", "RNN", "Transformer",]
        tk.Label(self.model_config_frame, text="Select Model:").grid(row=1, column=0, padx=10, pady=1)
        self.MODEL_NAME_var = tk.StringVar()
        self.MODEL_NAME_dropdown = ttk.Combobox(self.model_config_frame, textvariable=self.MODEL_NAME_var, values=self.MODEL_NAME_options)
        self.MODEL_NAME_dropdown.grid(row=1, column=1, padx=10, pady=1)
        self.MODEL_NAME_dropdown.current(0)

        separator = ttk.Separator(self.model_config_frame,  orient="horizontal")
        separator.grid(row=2, column=0, columnspan=4, padx=10, pady=1, sticky="ew")
        separator_label = ttk.Label(self.model_config_frame, text="Auto Hyper-parameters Tuning")
        separator_label.grid(row=2, column=0, columnspan=4, padx=10, pady=1)

        # TUNE_PARAMETERS
        self.TUNE_PARAMETERS_var = tk.BooleanVar(value=False)
        self.TUNE_PARAMETERS_checkbox = tk.Checkbutton(self.model_config_frame, text="Tune Hyper-parameters", variable=self.TUNE_PARAMETERS_var, command=self.base_on_TUNE_PARAMETERS)
        self.TUNE_PARAMETERS_checkbox.grid(row=3, column=0, padx=10, pady=1)

        # NUM_TRIALS
        tk.Label(self.model_config_frame, text="Number of Trials:").grid(row=3, column=1, padx=10, pady=1)
        self.NUM_TRIALS_var = tk.IntVar()
        self.NUM_TRIALS_entry = tk.Entry(self.model_config_frame, textvariable=self.NUM_TRIALS_var)
        self.NUM_TRIALS_entry.grid(row=3, column=2)
        self.NUM_TRIALS_var.set(20)

        ####################################
        # Optimization Layer Configuration #
        ####################################
        self.optimization_config_frame = tk.LabelFrame(self.root, text="Optimization Layer Configuration", font=("Helvetica", 10, "bold"))
        self.optimization_config_frame.pack(padx=10, pady=1, fill="none")

        # ttk.Label(self.optimization_config_frame, text=" "*150).grid(row=0, column=0, columnspan=4, padx=10, pady=0)
        # ttk.Label(self.optimization_config_frame, text=" "*150).grid(row=1000, column=0, columnspan=4, padx=10, pady=0)

        # NUM_EPOCHES
        tk.Label(self.optimization_config_frame, text="Number of Epoches:").grid(row=1, column=0, padx=10, pady=1)
        self.NUM_EPOCHES_var = tk.IntVar()
        self.NUM_EPOCHES_entry = tk.Entry(self.optimization_config_frame, textvariable=self.NUM_EPOCHES_var)
        self.NUM_EPOCHES_entry.grid(row=1, column=1, padx=10, pady=1)
        self.NUM_EPOCHES_var.set(100)

        # SAVE_EVERY
        tk.Label(self.optimization_config_frame, text="Save Every:").grid(row=1, column=2, padx=10, pady=1)
        self.SAVE_EVERY_var = tk.IntVar()
        self.SAVE_EVERY_entry = tk.Entry(self.optimization_config_frame, textvariable=self.SAVE_EVERY_var)
        self.SAVE_EVERY_entry.grid(row=1, column=3, padx=10, pady=1)
        self.SAVE_EVERY_var.set(1)

        separator = ttk.Separator(self.optimization_config_frame,  orient="horizontal")
        separator.grid(row=2, column=0, columnspan=4, padx=10, pady=1, sticky="ew")
        separator_label = ttk.Label(self.optimization_config_frame, text="Optimizer Settings")
        separator_label.grid(row=2, column=0, columnspan=4, padx=10, pady=1)

        # OPTIMIZER_NAME
        self.OPTIMIZER_NAME_options = ["Adadelta", "Adagrad", "Adam", "AdamW", "Adamax", "ASGD", "SGD", "RAdam", "Rprop",
                                       "RMSprop", "NAdam", "A2GradExp", "A2GradInc", "A2GradUni", "AccSGD", "AdaBelief",
                                       "AdaBound", "AdaMod", "Adafactor", "AdamP", "AggMo", "Apollo", "DiffGrad", "LARS",
                                       "Lamb", "MADGRAD", "NovoGrad", "PID", "QHAdam", "QHM", "Ranger", "RangerQH", "RangerVA",
                                       "SGDP", "SGDW", "SWATS", "Yogi"]
        tk.Label(self.optimization_config_frame, text="Select Optimizer:").grid(row=3, column=0, padx=10, pady=1)
        self.OPTIMIZER_NAME_var = tk.StringVar()
        self.OPTIMIZER_NAME_dropdown = ttk.Combobox(self.optimization_config_frame, textvariable=self.OPTIMIZER_NAME_var, values=self.OPTIMIZER_NAME_options)
        self.OPTIMIZER_NAME_dropdown.grid(row=3, column=1, padx=10, pady=1)
        self.OPTIMIZER_NAME_dropdown.current(2)

        # LEARNING_RATE
        tk.Label(self.optimization_config_frame, text="Set Learning Rate:").grid(row=3, column=2, padx=10, pady=1)
        self.LEARNING_RATE_var = tk.DoubleVar()
        self.LEARNING_RATE_entry = tk.Entry(self.optimization_config_frame, textvariable=self.LEARNING_RATE_var)
        self.LEARNING_RATE_entry.grid(row=3, column=3, padx=10, pady=1)
        self.LEARNING_RATE_var.set(0.001)

        separator = ttk.Separator(self.optimization_config_frame,  orient="horizontal")
        separator.grid(row=4, column=0, columnspan=4, padx=10, pady=1, sticky="ew")
        separator_label = ttk.Label(self.optimization_config_frame, text="Criterion Settings")
        separator_label.grid(row=4, column=0, columnspan=4, padx=10, pady=1)

        # CRITERION_NAME
        self.CRITERION_NAME_options = ["LOG_WEALTH", "LOG_WEALTH_L2_DIVERSIFICATION", "LOG_WEALTH_L2_CONCENTRATION",
                                       "L2_DIVERSIFICATION", "L2_CONCENTRATION", "SHARPE_RATIO", "VOLATILITY"]
        tk.Label(self.optimization_config_frame, text="Select Criterion:").grid(row=6, column=0, padx=10, pady=1)
        self.CRITERION_NAME_var = tk.StringVar()
        self.CRITERION_NAME_dropdown = ttk.Combobox(self.optimization_config_frame, textvariable=self.CRITERION_NAME_var, values=self.CRITERION_NAME_options)
        self.CRITERION_NAME_dropdown.grid(row=6, column=1, padx=10, pady=1)
        self.CRITERION_NAME_dropdown.current(0)

        # LAMBDA_L2
        tk.Label(self.optimization_config_frame, text="Set Lambda:").grid(row=6, column=2, padx=10, pady=1)
        self.LAMBDA_L2_var = tk.DoubleVar()
        self.LAMBDA_L2_entry = tk.Entry(self.optimization_config_frame, textvariable=self.LAMBDA_L2_var)
        self.LAMBDA_L2_entry.grid(row=6, column=3, padx=10, pady=1)
        self.LAMBDA_L2_var.set(0.0005)

        # Trace changes to CRITERION_NAME_var
        self.CRITERION_NAME_var.trace("w", self.base_on_CRITERION_NAME)
        # Initial state update
        self.base_on_CRITERION_NAME()

        # train_button
        self.train_button = tk.Button(self.root, text="Train Model", command=self.train_model)
        self.train_button.pack(padx=10, pady=3)

        ##################################
        # Evaluation Layer Configuration #
        ##################################
        self.evaluation_config_frame = tk.LabelFrame(self.root, text="Evaluation Layer Configuration", font=("Helvetica", 10, "bold"))
        self.evaluation_config_frame.pack(padx=10, pady=1, fill="none")

        # PLOT_CHINESE
        self.PLOT_CHINESE_var = tk.BooleanVar(value=False)
        self.PLOT_CHINESE_checkbox = tk.Checkbutton(self.evaluation_config_frame, text="Plot Chinese", variable=self.PLOT_CHINESE_var)
        self.PLOT_CHINESE_checkbox.grid(row=0, column=0, padx=10, pady=1)

        # PROP_WINNERS
        tk.Label(self.evaluation_config_frame, text="Proportion of Winners").grid(row=0, column=1, padx=10, pady=1)
        self.PROP_WINNERS_var = tk.DoubleVar()
        self.PROP_WINNERS_entry = tk.Entry(self.evaluation_config_frame, textvariable=self.PROP_WINNERS_var)
        self.PROP_WINNERS_entry.grid(row=0, column=2, padx=10, pady=1)
        self.PROP_WINNERS_var.set(0.5)

        separator = ttk.Separator(self.evaluation_config_frame,  orient="horizontal")
        separator.grid(row=1, column=0, columnspan=4, padx=10, pady=1, sticky="ew")
        separator_label = ttk.Label(self.evaluation_config_frame, text="Interpretability Analysis")
        separator_label.grid(row=1, column=0, columnspan=4, padx=10, pady=1)

        # INCLUDE_INTERPRETABILITY_ANALYSIS
        self.INCLUDE_INTERPRETABILITY_ANALYSIS_var = tk.BooleanVar(value=False)
        self.INCLUDE_INTERPRETABILITY_ANALYSIS_checkbox = tk.Checkbutton(self.evaluation_config_frame, text="Include Interpretability Analysis", variable=self.INCLUDE_INTERPRETABILITY_ANALYSIS_var)
        self.INCLUDE_INTERPRETABILITY_ANALYSIS_checkbox.grid(row=2, column=0, padx=10, pady=1)

        # INCLUDE_ECONOMIC_DISTILLATION
        self.INCLUDE_ECONOMIC_DISTILLATION_var = tk.BooleanVar(value=False)
        self.INCLUDE_ECONOMIC_DISTILLATION_checkbox = tk.Checkbutton(self.evaluation_config_frame, text="Include Economic Distillation", variable=self.INCLUDE_ECONOMIC_DISTILLATION_var)
        self.INCLUDE_ECONOMIC_DISTILLATION_checkbox.grid(row=2, column=1, padx=10, pady=1)

        # PROP_DISTILLED_FEATURES
        tk.Label(self.evaluation_config_frame, text="Proportion of Distilled Features:").grid(row=2, column=2, padx=10, pady=1)
        self.PROP_DISTILLED_FEATURES_var = tk.DoubleVar()
        self.PROP_DISTILLED_FEATURES_entry = tk.Entry(self.evaluation_config_frame, textvariable=self.PROP_DISTILLED_FEATURES_var)
        self.PROP_DISTILLED_FEATURES_entry.grid(row=2, column=3, padx=10, pady=1)
        self.PROP_DISTILLED_FEATURES_var.set(0.7)

        # DISTILLER_NAME
        self.DISTILLER_NAME_options = ["LinearRegression", "Ridge", "RidgeCV", "SGDRegressor", "ElasticNet", "ElasticNetCV",
                                       "Lars", "LarsCV", "Lasso", "LassoCV", "LassoLars", "LassoLarsCV", "LassoLarsIC",
                                       "OrthogonalMatchingPursuit", "OrthogonalMatchingPursuitCV", "ARDRegression",
                                       "BayesianRidge", "HuberRegressor", "QuantileRegressor", "RANSACRegressor",
                                       "TheilSenRegressor", "PoissonRegressor", "TweedieRegressor", "GammaRegressor",
                                       "PassiveAggressiveRegressor"]
        tk.Label(self.evaluation_config_frame, text="Select Distiller:").grid(row=3, column=0, padx=10, pady=1)
        self.DISTILLER_NAME_var = tk.StringVar()
        self.DISTILLER_NAME_dropdown = ttk.Combobox(self.evaluation_config_frame, textvariable=self.DISTILLER_NAME_var, values=self.DISTILLER_NAME_options)
        self.DISTILLER_NAME_dropdown.grid(row=3, column=1, padx=10, pady=1)
        self.DISTILLER_NAME_dropdown.current(8)

        # Y_NAME
        self.Y_NAME_options = ["SCORES", "PORTFOLIOS"]
        tk.Label(self.evaluation_config_frame, text="Select Y Name:").grid(row=3, column=2, padx=10, pady=1)
        self.Y_NAME_var = tk.StringVar()
        self.Y_NAME_dropdown = ttk.Combobox(self.evaluation_config_frame, textvariable=self.Y_NAME_var, values=self.Y_NAME_options)
        self.Y_NAME_dropdown.grid(row=3, column=3, padx=10, pady=1)
        self.Y_NAME_dropdown.current(1)

        separator = ttk.Separator(self.evaluation_config_frame,  orient="horizontal")
        separator.grid(row=4, column=0, columnspan=4, padx=10, pady=1, sticky="ew")
        separator_label = ttk.Label(self.evaluation_config_frame, text="Metric Settings")
        separator_label.grid(row=4, column=0, columnspan=4, padx=10, pady=1)

        # TRANSACTIOS_COSTS_RATE
        tk.Label(self.evaluation_config_frame, text="Set Transactios Costs Rate:").grid(row=5, column=0, padx=10, pady=1)
        self.TRANSACTIOS_COSTS_RATE_var = tk.DoubleVar()
        self.TRANSACTIOS_COSTS_RATE_entry = tk.Entry(self.evaluation_config_frame, textvariable=self.TRANSACTIOS_COSTS_RATE_var)
        self.TRANSACTIOS_COSTS_RATE_entry.grid(row=5, column=1, padx=10, pady=1)
        self.TRANSACTIOS_COSTS_RATE_var.set(0.005)

        # TRANSACTIOS_COSTS_RATE_INTERVAL
        tk.Label(self.evaluation_config_frame, text="Set Transactios Costs Rate Interval:").grid(row=5, column=2, padx=10, pady=1)
        self.TRANSACTIOS_COSTS_RATE_INTERVAL_var = tk.DoubleVar()
        self.TRANSACTIOS_COSTS_RATE_INTERVAL_entry = tk.Entry(self.evaluation_config_frame, textvariable=self.TRANSACTIOS_COSTS_RATE_INTERVAL_var)
        self.TRANSACTIOS_COSTS_RATE_INTERVAL_entry.grid(row=5, column=3, padx=10, pady=1)
        self.TRANSACTIOS_COSTS_RATE_INTERVAL_var.set(0.001)


        self.evaluate_button = tk.Button(self.root, text="Evaluate Model", command=self.evaluate_model)
        self.evaluate_button.pack(padx=10, pady=5)

        # self.quit_button = tk.Button(self.root, text="Quit", command=self.root.quit)
        # self.quit_button.pack(padx=10, pady=1)

        # self.result_frame = tk.LabelFrame(self.root, text="Results")
        # self.result_frame.pack(padx=10, pady=1, fill="both", expand=True)

    def base_on_CRITERION_NAME(self, *args):
        # Enable or disable the text box based on the selected value
        selected_value = self.CRITERION_NAME_var.get()
        if selected_value == "LOG_WEALTH_L2_DIVERSIFICATION" or selected_value == "LOG_WEALTH_L2_CONCENTRATION":
            self.LAMBDA_L2_entry.config(state="normal")
        else:
            self.LAMBDA_L2_entry.config(state="disabled")

    def base_on_LOAD_LOCAL_DATALOADER(self):
        if self.LOAD_LOCAL_DATALOADER_var.get():
            self.SCALER_dropdown.config(state="disabled")
            self.INCLUDE_OHLCV_FEATURES_chk.config(state="disabled")
            self.INCLUDE_OVERLAP_FEATURES_chk.config(state="disabled")
            self.INCLUDE_MOMENTUM_FEATURES_chk.config(state="disabled")
            self.INCLUDE_VOLUME_FEATURES_chk.config(state="disabled")
            self.INCLUDE_CYCLE_FEATURES_chk.config(state="disabled")
            self.INCLUDE_PRICE_FEATURES_chk.config(state="disabled")
            self.INCLUDE_VOLATILITY_FEATURES_chk.config(state="disabled")
            self.INCLUDE_PATTERN_FEATURES_chk.config(state="disabled")
            self.INCLUDE_WINDOW_DATA_chk.config(state="disabled")
            self.WINDOW_SIZE_entry.config(state="disabled")
        else:
            self.SCALER_dropdown.config(state="normal")
            self.INCLUDE_OHLCV_FEATURES_chk.config(state="normal")
            self.INCLUDE_OVERLAP_FEATURES_chk.config(state="normal")
            self.INCLUDE_MOMENTUM_FEATURES_chk.config(state="normal")
            self.INCLUDE_VOLUME_FEATURES_chk.config(state="normal")
            self.INCLUDE_CYCLE_FEATURES_chk.config(state="normal")
            self.INCLUDE_PRICE_FEATURES_chk.config(state="normal")
            self.INCLUDE_VOLATILITY_FEATURES_chk.config(state="normal")
            self.INCLUDE_PATTERN_FEATURES_chk.config(state="normal")
            self.INCLUDE_WINDOW_DATA_chk.config(state="normal")
            self.WINDOW_SIZE_entry.config(state="normal")

    def base_on_TUNE_PARAMETERS(self):
        if self.TUNE_PARAMETERS_var.get():
            self.NUM_TRIALS_entry.config(state="normal")
        else:
            self.NUM_TRIALS_entry.config(state="disabled")

    def base_on_INCLUDE_WINDOW_DATA(self):
        if self.INCLUDE_WINDOW_DATA_var.get():
            self.WINDOW_SIZE_entry.config(state="normal")
        else:
            self.WINDOW_SIZE_entry.config(state="disabled")

    def load_dataset(self):
        config = load_config()

        config["DEVICE"] = self.DEVICE_var.get()
        config["MANUAL_SEED"] = self.MANUAL_SEED_var.get()

        config["LOAD_LOCAL_DATALOADER"] = self.LOAD_LOCAL_DATALOADER_var.get()
        config["DATASET_NAME"] = self.DATASET_NAME_var.get()
        config["FEATURE_ENGINEERING_CONFIG"]["INCLUDE_OHLCV_FEATURES"] = self.INCLUDE_OHLCV_FEATURES_var.get()
        config["FEATURE_ENGINEERING_CONFIG"]["INCLUDE_OVERLAP_FEATURES"] = self.INCLUDE_OVERLAP_FEATURES_var.get()
        config["FEATURE_ENGINEERING_CONFIG"]["INCLUDE_MOMENTUM_FEATURES"] = self.INCLUDE_MOMENTUM_FEATURES_var.get()
        config["FEATURE_ENGINEERING_CONFIG"]["INCLUDE_VOLUME_FEATURES"] = self.INCLUDE_VOLUME_FEATURES_var.get()
        config["FEATURE_ENGINEERING_CONFIG"]["INCLUDE_CYCLE_FEATURES"] = self.INCLUDE_CYCLE_FEATURES_var.get()
        config["FEATURE_ENGINEERING_CONFIG"]["INCLUDE_PRICE_FEATURES"] = self.INCLUDE_PRICE_FEATURES_var.get()
        config["FEATURE_ENGINEERING_CONFIG"]["INCLUDE_VOLATILITY_FEATURES"] = self.INCLUDE_VOLATILITY_FEATURES_var.get()
        config["FEATURE_ENGINEERING_CONFIG"]["INCLUDE_PATTERN_FEATURES"] = self.INCLUDE_PATTERN_FEATURES_var.get()
        config["DATA_AUGMENTATION_CONFIG"]["WINDOW_DATA"]["INCLUDE_WINDOW_DATA"] = self.INCLUDE_WINDOW_DATA_var.get()
        config["DATA_AUGMENTATION_CONFIG"]["WINDOW_DATA"]["WINDOW_SIZE"] = self.WINDOW_SIZE_var.get()
        config["SCALER"] = self.SCALER_var.get()

        update_config(config)

        try:
            self.load_dataset_output = DatasetLoader().load_dataset()
            messagebox.showinfo("Success", f"Dataset '{self.DATASET_NAME_var.get()}' loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset '{self.DATASET_NAME_var.get()}': {e}")


    def train_model(self):
        config = load_config()

        config["MODEL_NAME"] = self.MODEL_NAME_var.get()
        config["TUNE_PARAMETERS"] = self.TUNE_PARAMETERS_var.get()
        config["NUM_TRIALS"] = self.NUM_TRIALS_var.get()

        config["NUM_EPOCHES"] = self.NUM_EPOCHES_var.get()
        config["SAVE_EVERY"] = self.SAVE_EVERY_var.get()
        config["OPTIMIZER_NAME"] = self.OPTIMIZER_NAME_var.get()
        config["LEARNING_RATE"] = self.LEARNING_RATE_var.get()
        config["CRITERION_NAME"] = self.CRITERION_NAME_var.get()
        config["LAMBDA_L2"] = self.LAMBDA_L2_var.get()

        update_config(config)

        if hasattr(self, 'load_dataset_output'):
            try:
                self.train_model_output = ModelTrainer(self.load_dataset_output).train_model()
                messagebox.showinfo("Success", f"Model '{self.MODEL_NAME_var.get()}' trained successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to train model '{self.MODEL_NAME_var.get()}': {e}")
        else:
            messagebox.showwarning("Warning", "Please load the dataset first!")

    def evaluate_model(self):
        config = load_config()

        config["PLOT_CHINESE"] = self.PLOT_CHINESE_var.get()
        config["PROP_WINNERS"] = self.PROP_WINNERS_var.get()
        config["INTERPRETABLE_ANALYSIS_CONFIG"]["INCLUDE_INTERPRETABILITY_ANALYSIS"] = self.INCLUDE_INTERPRETABILITY_ANALYSIS_var.get()
        config["INTERPRETABLE_ANALYSIS_CONFIG"]["INCLUDE_ECONOMIC_DISTILLATION"] = self.INCLUDE_ECONOMIC_DISTILLATION_var.get()
        config["INTERPRETABLE_ANALYSIS_CONFIG"]["PROP_DISTILLED_FEATURES"] = self.PROP_DISTILLED_FEATURES_var.get()
        config["INTERPRETABLE_ANALYSIS_CONFIG"]["DISTILLER_NAME"] = self.DISTILLER_NAME_var.get()
        config["INTERPRETABLE_ANALYSIS_CONFIG"]["Y_NAME"] = self.Y_NAME_var.get()
        config["METRIC_CONFIG"]["PRACTICAL_METRICS"]["TRANSACTIOS_COSTS_RATE"] = self.TRANSACTIOS_COSTS_RATE_var.get()
        config["METRIC_CONFIG"]["PRACTICAL_METRICS"]["TRANSACTIOS_COSTS_RATE_INTERVAL"] = self.TRANSACTIOS_COSTS_RATE_INTERVAL_var.get()

        update_config(config)

        if hasattr(self, 'load_dataset_output') and hasattr(self, 'train_model_output'):
            try:
                self.evaluate_model_output = ModelEvaluator(self.load_dataset_output, self.train_model_output).evaluate_model()
                messagebox.showinfo("Success", f"Model '{self.MODEL_NAME_var.get()}' evaluated successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to evaluate model: '{self.MODEL_NAME_var.get()}': {e}")
        else:
            messagebox.showwarning("Warning", "Please load the dataset and train the model first!")


if __name__ == "__main__":
    app = FinOLApp()
    app.run()