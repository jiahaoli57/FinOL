import time
import json
import tkinter as tk

from PIL import Image, ImageTk
from tkinter import messagebox, ttk
from finol.data_layer import DatasetLoader
from finol.optimization_layer import ModelTrainer
from finol.evaluation_layer import ModelEvaluator
from finol.utils import ROOT_PATH, load_config, update_config, detect_device


# def format_string(input_string):
#     words = input_string.split("_")
#     formatted_words = [word.capitalize() for word in words]
#     formatted_string = " ".join(formatted_words)
#
#     return formatted_string


class FinOLApp:
    def __init__(self):
        self.config = load_config()
        self.root = tk.Tk()

        # self.root.tk.call('source', 'forest-light.tcl')
        # ttk.Style().theme_use("winnative")  # alt classic

        # self.root.state("zoomed")
        self.root.iconphoto(False, ImageTk.PhotoImage(Image.open(ROOT_PATH + "/finol_logo_icon.png")))
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
        self.general_config_frame.pack(padx=100, pady=1, fill="none")

        # DEVICE
        options = ["auto", "cpu", "cuda"]
        self.create_dropdown(self.general_config_frame, options, "Select Device:", 0, 0, options.index(self.config["DEVICE"]), "StringVar", ["DEVICE"])
        # trace_dropdown with default value
        self.config = detect_device(self.config)
        update_config(self.config)

        # MANUAL_SEED
        self.create_entry(self.general_config_frame, "Set Seed:", 0, 2, self.config["MANUAL_SEED"], "IntVar", ["MANUAL_SEED"])

        ############################
        # Data Layer Configuration #
        ############################
        self.data_config_frame = tk.LabelFrame(self.root, text="Data Layer Configuration", font=("Helvetica", 10, "bold"))
        self.data_config_frame.pack(padx=10, pady=1, fill="none")

        # DATASET_NAME
        options = ["NYSE(O)", "NYSE(N)", "DJIA", "SP500", "TSE", "SSE", "HSI", "CMEG", "CRYPTO"]
        self.create_dropdown(self.data_config_frame, options, "Select Dataset:", 0, 0, options.index(self.config["DATASET_NAME"]), "StringVar", ["DATASET_NAME"])

        # SCALER
        options = ["None", "StandardScaler", "MinMaxScaler", "MaxAbsScaler", "RobustScaler", "WindowStandardScaler",
                   "WindowMinMaxScaler", "WindowMaxAbsScaler", "WindowRobustScaler"]
        self.create_dropdown(self.data_config_frame, options, "Select Scaler:", 0, 2, options.index(self.config["SCALER"]), "StringVar", ["SCALER"])

        # create_separator
        self.create_separator(frame=self.data_config_frame, row=1, text="Auto Feature Engineering")

        # INCLUDE_OHLCV_FEATURES
        self.create_checkbox(self.data_config_frame, "Include OHLCV Features", 2, 0, self.config["FEATURE_ENGINEERING_CONFIG"]["INCLUDE_OHLCV_FEATURES"], ["FEATURE_ENGINEERING_CONFIG", "INCLUDE_OHLCV_FEATURES"])

        # INCLUDE_OVERLAP_FEATURES
        self.create_checkbox(self.data_config_frame, "Include Overlap Features", 2, 1, self.config["FEATURE_ENGINEERING_CONFIG"]["INCLUDE_OVERLAP_FEATURES"], ["FEATURE_ENGINEERING_CONFIG", "INCLUDE_OVERLAP_FEATURES"])

        # INCLUDE_MOMENTUM_FEATURES
        self.create_checkbox(self.data_config_frame, "Include Momentum Features", 2, 2, self.config["FEATURE_ENGINEERING_CONFIG"]["INCLUDE_MOMENTUM_FEATURES"], ["FEATURE_ENGINEERING_CONFIG", "INCLUDE_MOMENTUM_FEATURES"])

        # INCLUDE_VOLUME_FEATURES
        self.create_checkbox(self.data_config_frame, "Include Volume Features", 2, 3, self.config["FEATURE_ENGINEERING_CONFIG"]["INCLUDE_VOLUME_FEATURES"], ["FEATURE_ENGINEERING_CONFIG", "INCLUDE_VOLUME_FEATURES"])

        # INCLUDE_CYCLE_FEATURES
        self.create_checkbox(self.data_config_frame, "Include Cycle Features", 3, 0, self.config["FEATURE_ENGINEERING_CONFIG"]["INCLUDE_CYCLE_FEATURES"], ["FEATURE_ENGINEERING_CONFIG", "INCLUDE_CYCLE_FEATURES"])

        #  INCLUDE_PRICE_FEATURES
        self.create_checkbox(self.data_config_frame, "Include Price Features", 3, 1, self.config["FEATURE_ENGINEERING_CONFIG"]["INCLUDE_PRICE_FEATURES"], ["FEATURE_ENGINEERING_CONFIG", "INCLUDE_PRICE_FEATURES"])

        # INCLUDE_VOLATILITY_FEATURES
        self.create_checkbox(self.data_config_frame, "Include Volatility Features", 3, 2, self.config["FEATURE_ENGINEERING_CONFIG"]["INCLUDE_VOLATILITY_FEATURES"], ["FEATURE_ENGINEERING_CONFIG", "INCLUDE_VOLATILITY_FEATURES"])

        #  INCLUDE_PATTERN_FEATURES
        self.create_checkbox(self.data_config_frame, "Include Pattern Features", 3, 3, self.config["FEATURE_ENGINEERING_CONFIG"]["INCLUDE_PATTERN_FEATURES"], ["FEATURE_ENGINEERING_CONFIG", "INCLUDE_PATTERN_FEATURES"])

        # create_separator
        self.create_separator(frame=self.data_config_frame, row=4, text="Data Augmentation")

        # INCLUDE_WINDOW_DATA
        self.create_checkbox(self.data_config_frame, "Include Window Data", 5, 0, self.config["DATA_AUGMENTATION_CONFIG"]["WINDOW_DATA"]["INCLUDE_WINDOW_DATA"], ["DATA_AUGMENTATION_CONFIG", "WINDOW_DATA", "INCLUDE_WINDOW_DATA"])

        # WINDOW_SIZE
        self.create_entry(self.data_config_frame, "Set Window Size:", 5, 1, self.config["DATA_AUGMENTATION_CONFIG"]["WINDOW_DATA"]["WINDOW_SIZE"], "IntVar", ["DATA_AUGMENTATION_CONFIG", "WINDOW_DATA", "WINDOW_SIZE"])
        # trace_checkbox with default value
        self.trace_checkbox(["DATA_AUGMENTATION_CONFIG", "WINDOW_DATA", "INCLUDE_WINDOW_DATA"], self.config["DATA_AUGMENTATION_CONFIG"]["WINDOW_DATA"]["INCLUDE_WINDOW_DATA"])

        # create_separator
        self.create_separator(frame=self.data_config_frame, row=6)

        # LOAD_LOCAL_DATALOADER
        self.create_checkbox(self.data_config_frame, "Load Local Dataloader", 7, 0, self.config["LOAD_LOCAL_DATALOADER"], ["LOAD_LOCAL_DATALOADER"])
        # trace_checkbox with default value
        self.trace_checkbox(["LOAD_LOCAL_DATALOADER"], self.config["LOAD_LOCAL_DATALOADER"])

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
        options = ["--", "AlphaPortfolio", "CNN", "DNN", "LSRE-CAAN", "LSTM", "RNN", "Transformer",]
        self.create_dropdown(self.model_config_frame, options, "Select Model:", 0, 2, options.index(self.config["MODEL_NAME"]), "StringVar", ["MODEL_NAME"])

        # create_separator
        self.create_separator(frame=self.model_config_frame, row=1, text="Model Parameters")

        ####################################
        # Optimization Layer Configuration #
        ####################################
        self.optimization_config_frame = tk.LabelFrame(self.root, text="Optimization Layer Configuration", font=("Helvetica", 10, "bold"))
        self.optimization_config_frame.pack(padx=10, pady=1, fill="none")

        # NUM_EPOCHES
        self.create_entry(self.optimization_config_frame, "Number of Epoches:", 1, 0, self.config["NUM_EPOCHES"], "IntVar", ["NUM_EPOCHES"])

        # SAVE_EVERY
        self.create_entry(self.optimization_config_frame, "Save Every:", 1, 2, self.config["SAVE_EVERY"], "IntVar", ["SAVE_EVERY"])

        # create_separator
        self.create_separator(frame=self.optimization_config_frame, row=2, text="Optimizer Settings")

        # OPTIMIZER_NAME
        options = ["Adadelta", "Adagrad", "Adam", "AdamW", "Adamax", "ASGD", "SGD", "RAdam", "Rprop", "RMSprop",
                   "NAdam", "A2GradExp", "A2GradInc", "A2GradUni", "AccSGD", "AdaBelief", "AdaBound", "AdaMod",
                   "Adafactor", "AdamP", "AggMo", "Apollo", "DiffGrad", "LARS", "Lamb", "MADGRAD", "NovoGrad", "PID",
                   "QHAdam", "QHM", "Ranger", "RangerQH", "RangerVA", "SGDP", "SGDW", "SWATS", "Yogi"]
        self.create_dropdown(self.optimization_config_frame, options, "Select Optimizer:", 3, 0, options.index(self.config["OPTIMIZER_NAME"]), "StringVar", ["OPTIMIZER_NAME"])

        # LEARNING_RATE
        self.create_entry(self.optimization_config_frame, "Set Learning Rate:", 3, 2, self.config["LEARNING_RATE"], "DoubleVar", ["LEARNING_RATE"])

        # create_separator
        self.create_separator(frame=self.optimization_config_frame, row=4, text="Criterion Settings")

        # CRITERION_NAME
        options = ["LogWealth", "LogWealth_L2Diversification", "LogWealth_L2Concentration", "L2Diversification", "L2Concentration", "SharpeRatio", "Volatility"]
        self.create_dropdown(self.optimization_config_frame, options, "Select Criterion:", 6, 0, options.index(self.config["CRITERION_NAME"]), "StringVar", ["CRITERION_NAME"])

        # LAMBDA_L2
        self.create_entry(self.optimization_config_frame, "Set Lambda:", 6, 2, self.config["LAMBDA_L2"], "DoubleVar", ["LAMBDA_L2"])
        # trace_dropdown with default value
        self.trace_dropdown(["CRITERION_NAME"], self.config["CRITERION_NAME"])

        # create_separator
        self.create_separator(frame=self.optimization_config_frame, row=100, text="Auto Hyper-parameters Tuning")

        # NUM_TRIALS
        self.create_entry(self.optimization_config_frame, "Number of Trials:", 101, 0, self.config["NUM_TRIALS"], "IntVar", ["NUM_TRIALS"])

        # SAMPLER_NAME
        options = ["BruteForceSampler", "CmaEsSampler", "GridSampler", "NSGAIISampler", "NSGAIIISampler", "QMCSampler",
                   "RandomSampler", "TPESampler", "GPSampler",]
        self.create_dropdown(self.optimization_config_frame, options, "Select Sampler:", 101, 2, options.index(self.config["SAMPLER_NAME"]), "StringVar", ["SAMPLER_NAME"])

        # PRUNER_NAME
        options = ["HyperbandPruner", "MedianPruner", "NopPruner", "PatientPruner", "SuccessiveHalvingPruner", "WilcoxonPruner"]
        self.create_dropdown(self.optimization_config_frame, options, "Select Pruner:", 102, 0, options.index(self.config["PRUNER_NAME"]), "StringVar", ["PRUNER_NAME"])

        # WRAPPED_PRUNER_NAME
        options = ["HyperbandPruner", "MedianPruner", "SuccessiveHalvingPruner", "WilcoxonPruner"]
        self.create_dropdown(self.optimization_config_frame, options, "Select Wrapped Pruner:", 102, 2, options.index(self.config["WRAPPED_PRUNER_NAME"]), "StringVar", ["WRAPPED_PRUNER_NAME"])
        # trace_dropdown with default value
        self.trace_dropdown(["PRUNER_NAME"], self.config["PRUNER_NAME"])

        # TUNE_PARAMETERS
        self.create_checkbox(self.optimization_config_frame, "Tune Hyper-parameters", 103, 0, self.config["TUNE_PARAMETERS"], ["TUNE_PARAMETERS"])
        # trace_checkbox with default value
        self.trace_checkbox(["TUNE_PARAMETERS"], self.config["TUNE_PARAMETERS"])

        # train_button
        self.train_button = tk.Button(self.root, text="Train Model", command=self.train_model)
        self.train_button.pack(padx=10, pady=3)

        ##################################
        # Evaluation Layer Configuration #
        ##################################
        self.evaluation_config_frame = tk.LabelFrame(self.root, text="Evaluation Layer Configuration", font=("Helvetica", 10, "bold"))
        self.evaluation_config_frame.pack(padx=10, pady=1, fill="none")

        # PLOT_LANGUAGE
        # self.PLOT_CHINESE_var = tk.BooleanVar(value=False)
        # self.PLOT_CHINESE_checkbox = tk.Checkbutton(self.evaluation_config_frame, text="Plot Chinese", variable=self.PLOT_CHINESE_var)
        # self.PLOT_CHINESE_checkbox.grid(row=0, column=0, padx=10, pady=1)
        options = ["EN", "CN"]
        self.create_dropdown(self.evaluation_config_frame, options, "Select Plot Language:", 0, 0, options.index(self.config["PLOT_LANGUAGE"]), "StringVar", ["PLOT_LANGUAGE"])

        # PROP_WINNERS
        self.create_entry(self.evaluation_config_frame, "Proportion of Winners:", 0, 2, self.config["PROP_WINNERS"], "DoubleVar", ["PROP_WINNERS"])

        # create_separator
        self.create_separator(frame=self.evaluation_config_frame, row=1, text="Interpretability Analysis")

        # INCLUDE_INTERPRETABILITY_ANALYSIS
        self.create_checkbox(self.evaluation_config_frame, "Include Interpretability Analysis", 2, 0, self.config["INTERPRETABLE_ANALYSIS_CONFIG"]["INCLUDE_INTERPRETABILITY_ANALYSIS"], ["INTERPRETABLE_ANALYSIS_CONFIG", "INCLUDE_INTERPRETABILITY_ANALYSIS"])

        # INCLUDE_ECONOMIC_DISTILLATION
        self.create_checkbox(self.evaluation_config_frame, "Include Economic Distillation", 2, 1, self.config["INTERPRETABLE_ANALYSIS_CONFIG"]["INCLUDE_ECONOMIC_DISTILLATION"], ["INTERPRETABLE_ANALYSIS_CONFIG", "INCLUDE_ECONOMIC_DISTILLATION"])

        # PROP_DISTILLED_FEATURES
        self.create_entry(self.evaluation_config_frame, "Proportion of Distilled Features:", 2, 2, self.config["INTERPRETABLE_ANALYSIS_CONFIG"]["PROP_DISTILLED_FEATURES"], "DoubleVar", ["INTERPRETABLE_ANALYSIS_CONFIG", "PROP_DISTILLED_FEATURES"])

        # DISTILLER_NAME
        options = ["LinearRegression", "Ridge", "RidgeCV", "SGDRegressor", "ElasticNet", "ElasticNetCV", "Lars",
                   "LarsCV", "Lasso", "LassoCV", "LassoLars", "LassoLarsCV", "LassoLarsIC", "OrthogonalMatchingPursuit",
                   "OrthogonalMatchingPursuitCV", "ARDRegression", "BayesianRidge", "HuberRegressor", "QuantileRegressor",
                   "RANSACRegressor", "TheilSenRegressor", "PoissonRegressor", "TweedieRegressor", "GammaRegressor",
                   "PassiveAggressiveRegressor"]
        self.create_dropdown(self.evaluation_config_frame, options, "Select Distiller:", 3, 0, options.index(self.config["INTERPRETABLE_ANALYSIS_CONFIG"]["DISTILLER_NAME"]), "StringVar", ["INTERPRETABLE_ANALYSIS_CONFIG", "DISTILLER_NAME"])

        # Y_NAME
        options = ["Scores", "Portfolios"]
        self.create_dropdown(self.evaluation_config_frame, options, "Select Y Name:", 3, 2, options.index(self.config["INTERPRETABLE_ANALYSIS_CONFIG"]["Y_NAME"]), "StringVar", ["INTERPRETABLE_ANALYSIS_CONFIG", "Y_NAME"])

        # separator
        self.create_separator(frame=self.evaluation_config_frame, row=4, text="Metric Settings")

        # TRANSACTIOS_COSTS_RATE
        self.create_entry(self.evaluation_config_frame, "Set Transactios Costs Rate:", 5, 0, self.config["METRIC_CONFIG"]["PRACTICAL_METRICS"]["TRANSACTIOS_COSTS_RATE"], "DoubleVar", ["METRIC_CONFIG", "PRACTICAL_METRICS", "TRANSACTIOS_COSTS_RATE"])

        # TRANSACTIOS_COSTS_RATE_INTERVAL
        self.create_entry(self.evaluation_config_frame, "Set Transactios Costs Rate Interval:", 5, 2, self.config["METRIC_CONFIG"]["PRACTICAL_METRICS"]["TRANSACTIOS_COSTS_RATE_INTERVAL"], "DoubleVar", ["METRIC_CONFIG", "PRACTICAL_METRICS", "TRANSACTIOS_COSTS_RATE_INTERVAL"])

        self.evaluate_button = tk.Button(self.root, text="Evaluate Model", command=self.evaluate_model)
        self.evaluate_button.pack(padx=10, pady=5)

        # self.quit_button = tk.Button(self.root, text="Quit", command=self.root.quit)
        # self.quit_button.pack(padx=10, pady=1)

        # self.result_frame = tk.LabelFrame(self.root, text="Results")
        # self.result_frame.pack(padx=10, pady=1, fill="both", expand=True)

    def create_checkbox(self, frame, text, row, column, default_value, arg_name):
        pady = 1
        var = tk.BooleanVar(value=default_value)
        checkbox = tk.Checkbutton(frame, text=text, variable=var)
        checkbox.grid(row=row, column=column, padx=10, pady=pady)

        checkbox_mapping = {
            "INCLUDE_OHLCV_FEATURES": "INCLUDE_OHLCV_FEATURES_checkbox",
            "INCLUDE_OVERLAP_FEATURES": "INCLUDE_OVERLAP_FEATURES_checkbox",
            "INCLUDE_MOMENTUM_FEATURES": "INCLUDE_MOMENTUM_FEATURES_checkbox",
            "INCLUDE_VOLUME_FEATURES": "INCLUDE_VOLUME_FEATURES_checkbox",
            "INCLUDE_CYCLE_FEATURES": "INCLUDE_CYCLE_FEATURES_checkbox",
            "INCLUDE_PRICE_FEATURES": "INCLUDE_PRICE_FEATURES_checkbox",
            "INCLUDE_VOLATILITY_FEATURES": "INCLUDE_VOLATILITY_FEATURES_checkbox",
            "INCLUDE_PATTERN_FEATURES": "INCLUDE_PATTERN_FEATURES_checkbox",
            "INCLUDE_WINDOW_DATA": "INCLUDE_WINDOW_DATA_checkbox",
            "TUNE_PARAMETERS": "TUNE_PARAMETERS_checkbox",
        }
        for key, attr in checkbox_mapping.items():
            if key in arg_name:
                setattr(self, attr, checkbox)
                break

        # write default value to config
        self.write_var_to_config(arg_name, var.get())

        # trace var change, write new var value to config
        var.trace("w", lambda *args: self.write_var_to_config(arg_name, var.get()))

        # trace var change, modify others state
        var.trace("w", lambda *args: self.trace_checkbox(arg_name, var.get()))

    def trace_checkbox(self, arg_name, var_value):
        if "LOAD_LOCAL_DATALOADER" in arg_name:
            widgets = [
                self.SCALER_dropdown,
                self.INCLUDE_WINDOW_DATA_checkbox,
                self.INCLUDE_OHLCV_FEATURES_checkbox,
                self.INCLUDE_OVERLAP_FEATURES_checkbox,
                self.INCLUDE_MOMENTUM_FEATURES_checkbox,
                self.INCLUDE_VOLUME_FEATURES_checkbox,
                self.INCLUDE_CYCLE_FEATURES_checkbox,
                self.INCLUDE_PRICE_FEATURES_checkbox,
                self.INCLUDE_VOLATILITY_FEATURES_checkbox,
                self.INCLUDE_PATTERN_FEATURES_checkbox,
                self.WINDOW_SIZE_entry,
            ]
            state = "disabled" if var_value else "normal"
            for widget in widgets:
                widget.config(state=state)

            # double check
            if self.config["DATA_AUGMENTATION_CONFIG"]["WINDOW_DATA"]["INCLUDE_WINDOW_DATA"] is False:
                self.WINDOW_SIZE_entry.config(state="disabled")

        if "TUNE_PARAMETERS" in arg_name:
            widgets = [
                self.NUM_TRIALS_entry,
                self.SAMPLER_NAME_dropdown,
                self.PRUNER_NAME_dropdown,
                self.WRAPPED_PRUNER_NAME_dropdown,
            ]
            state = "normal" if var_value else "disabled"
            for widget in widgets:
                widget.config(state=state)

        if "INCLUDE_WINDOW_DATA" in arg_name:
            widgets = [self.WINDOW_SIZE_entry,]
            state = "normal" if var_value else "disabled"
            for widget in widgets:
                widget.config(state=state)

    def create_dropdown(self, frame, options, text, row, column, default_value, value_type, arg_name):
        pady = 1
        tk.Label(frame, text=text).grid(row=row, column=column, padx=10, pady=pady)
        var = getattr(tk, value_type)()
        dropdown = ttk.Combobox(frame, textvariable=var, values=options)
        dropdown.grid(row=row, column=column+1, padx=10, pady=pady)
        dropdown.current(default_value)

        dropdown_mapping = {
            "SCALER": "SCALER_dropdown",
            "MODEL_NAME": "MODEL_NAME_dropdown",
            "SAMPLER_NAME": "SAMPLER_NAME_dropdown",
            "PRUNER_NAME": "PRUNER_NAME_dropdown",
            "WRAPPED_PRUNER_NAME": "WRAPPED_PRUNER_NAME_dropdown",
        }
        for key, attr in dropdown_mapping.items():
            if key in arg_name:
                setattr(self, attr, dropdown)
                break

        # write default value to config
        self.write_var_to_config(arg_name, var.get())

        # trace var change, write new var value to config
        var.trace("w", lambda *args: self.write_var_to_config(arg_name, var.get()))

        # trace var change, modify others state
        var.trace("w", lambda *args: self.trace_dropdown(arg_name, var.get()))

    def trace_dropdown(self, arg_name, var_value):
        if "DEVICE" in arg_name:
            self.config = detect_device(self.config)
            update_config(self.config)

        if "CRITERION_NAME" in arg_name:
            widgets = [self.LAMBDA_L2_entry,]
            state = "normal" if var_value == "LogWealth_L2Diversification" or var_value == "LogWealth_L2Concentration" else "disabled"
            for widget in widgets:
                widget.config(state=state)

        if "MODEL_NAME" in arg_name:
            # reload the self.config as the config might change when running the optuna_optimizer.py
            self.config = load_config()
            #
            MODEL_NAME = var_value
            for widget in self.model_config_frame.winfo_children():
                if str(widget) in [".!labelframe3.!label", ".!labelframe3.!combobox",]:
                    pass
                else:
                    widget.destroy()

            # write default value to config
            self.write_var_to_config(["MODEL_NAME"], MODEL_NAME)

            # create_separator
            self.create_separator(frame=self.model_config_frame, row=2, text="Model Parameters")

            default_model_parms = self.config["MODEL_PARAMS"][MODEL_NAME]
            if MODEL_NAME == "AlphaPortfolio":
                self.create_entry(self.model_config_frame, "Number of Layers:", row=3, column=0, default_value=default_model_parms["NUM_LAYERS"],
                                  value_type="IntVar", arg_name=["MODEL_PARAMS", MODEL_NAME, "NUM_LAYERS"])
                self.create_entry(self.model_config_frame, "Dimension of Embedding:", row=3, column=2, default_value=default_model_parms["DIM_EMBEDDING"],
                                  value_type="IntVar", arg_name=["MODEL_PARAMS", MODEL_NAME, "DIM_EMBEDDING"])
                self.create_entry(self.model_config_frame, "Dimension of Feedforward:", row=3, column=4, default_value=default_model_parms["DIM_FEEDFORWARD"],
                                  value_type="IntVar", arg_name=["MODEL_PARAMS", MODEL_NAME, "DIM_FEEDFORWARD"])
                self.create_entry(self.model_config_frame, "Number of Heads:", row=4, column=0, default_value=default_model_parms["NUM_HEADS"],
                                  value_type="IntVar", arg_name=["MODEL_PARAMS", MODEL_NAME, "NUM_HEADS"])
                self.create_entry(self.model_config_frame, "Dropout Rate:", row=4, column=2, default_value=default_model_parms["DROPOUT"],
                                  value_type="DoubleVar", arg_name=["MODEL_PARAMS", MODEL_NAME, "DROPOUT"])

            elif MODEL_NAME == "CNN":
                self.create_entry(self.model_config_frame, "Out Channels:", row=3, column=0, default_value=default_model_parms["OUT_CHANNELS"],
                                  value_type="IntVar", arg_name=["MODEL_PARAMS", MODEL_NAME, "OUT_CHANNELS"])
                self.create_entry(self.model_config_frame, "Kernel Size:", row=3, column=2, default_value=default_model_parms["KERNEL_SIZE"],
                                  value_type="IntVar", arg_name=["MODEL_PARAMS", MODEL_NAME, "KERNEL_SIZE"])
                self.create_entry(self.model_config_frame, "Stride:", row=3, column=4, default_value=default_model_parms["STRIDE"],
                                  value_type="IntVar", arg_name=["MODEL_PARAMS", MODEL_NAME, "STRIDE"])
                self.create_entry(self.model_config_frame, "Hidden Size:", row=4, column=0, default_value=default_model_parms["HIDDEN_SIZE"],
                                  value_type="IntVar", arg_name=["MODEL_PARAMS", MODEL_NAME, "HIDDEN_SIZE"])
                self.create_entry(self.model_config_frame, "Dropout Rate:", row=4, column=2, default_value=default_model_parms["DROPOUT"],
                                  value_type="DoubleVar", arg_name=["MODEL_PARAMS", MODEL_NAME, "DROPOUT"])

            elif MODEL_NAME == "DNN":
                self.create_entry(self.model_config_frame, "Number of Layers:", row=3, column=0, default_value=default_model_parms["NUM_LAYERS"],
                                  value_type="IntVar", arg_name=["MODEL_PARAMS", MODEL_NAME, "NUM_LAYERS"])
                self.create_entry(self.model_config_frame, "Hidden Size:", row=3, column=2, default_value=default_model_parms["HIDDEN_SIZE"],
                                  value_type="IntVar", arg_name=["MODEL_PARAMS", MODEL_NAME, "HIDDEN_SIZE"])
                self.create_entry(self.model_config_frame, "Dropout Rate:", row=3, column=4, default_value=default_model_parms["DROPOUT"],
                                  value_type="DoubleVar", arg_name=["MODEL_PARAMS", MODEL_NAME, "DROPOUT"])

            elif MODEL_NAME == "LSRE-CAAN":
                self.create_entry(self.model_config_frame, "Number of Layers:", row=3, column=0, default_value=default_model_parms["NUM_LAYERS"],
                                  value_type="IntVar", arg_name=["MODEL_PARAMS", MODEL_NAME, "NUM_LAYERS"])
                self.create_entry(self.model_config_frame, "Number of Latents:", row=3, column=2, default_value=default_model_parms["NUM_LATENTS"],
                                  value_type="IntVar", arg_name=["MODEL_PARAMS", MODEL_NAME, "NUM_LATENTS"])
                self.create_entry(self.model_config_frame, "Dimension of Latent:", row=3, column=4, default_value=default_model_parms["LATENT_DIM"],
                                  value_type="IntVar", arg_name=["MODEL_PARAMS", MODEL_NAME, "LATENT_DIM"])
                self.create_entry(self.model_config_frame, "Number of Cross Heads:", row=4, column=0, default_value=default_model_parms["CROSS_HEADS"],
                                  value_type="IntVar", arg_name=["MODEL_PARAMS", MODEL_NAME, "CROSS_HEADS"])
                self.create_entry(self.model_config_frame, "Number of Latent Heads:", row=4, column=2, default_value=default_model_parms["LATENT_HEADS"],
                                  value_type="IntVar", arg_name=["MODEL_PARAMS", MODEL_NAME, "LATENT_HEADS"])
                self.create_entry(self.model_config_frame, "Dimensions per Cross Head:", row=4, column=4, default_value=default_model_parms["CROSS_DIM_HEAD"],
                                  value_type="IntVar", arg_name=["MODEL_PARAMS", MODEL_NAME, "CROSS_DIM_HEAD"])
                self.create_entry(self.model_config_frame, "Dimensions per Latent Head:", row=5, column=0, default_value=default_model_parms["LATENT_DIM_HEAD"],
                                  value_type="IntVar", arg_name=["MODEL_PARAMS", MODEL_NAME, "LATENT_DIM_HEAD"])
                self.create_entry(self.model_config_frame, "Dropout Rate:", row=5, column=2, default_value=default_model_parms["DROPOUT"],
                                  value_type="DoubleVar", arg_name=["MODEL_PARAMS", MODEL_NAME, "DROPOUT"])


            # # trace var change, write new var value to config
            # self.MODEL_NAME_var.trace("w", lambda *args: self.write_var_to_config(["MODEL_NAME"], MODEL_NAME))

        if "PRUNER_NAME" in arg_name:
            widgets = [self.WRAPPED_PRUNER_NAME_dropdown,]
            state = "normal" if var_value == "PatientPruner" else "disabled"
            for widget in widgets:
                widget.config(state=state)


    def create_separator(self, frame, row, text=None):
        # separator = ttk.Separator(frame, orient="horizontal")
        # separator.grid(row=row, column=0, columnspan=6, padx=10, pady=1, sticky="ew")
        separator_label = ttk.Label(frame, text="-"*210)
        separator_label.grid(row=row, column=0, columnspan=6, padx=10, pady=1)
        if text != None:
            separator_label = ttk.Label(frame, text=text)
            separator_label.grid(row=row, column=0, columnspan=6, padx=10, pady=1)

    def create_entry(self, frame, text, row, column, default_value, value_type, arg_name):
        pady = 1
        tk.Label(frame, text=text).grid(row=row, column=column, padx=10, pady=pady)
        var = getattr(tk, value_type)()
        entry = tk.Entry(frame, textvariable=var)
        entry.grid(row=row, column=column + 1, padx=10, pady=pady)
        var.set(default_value)

        entry_mapping = {
            "WINDOW_SIZE": "WINDOW_SIZE_entry",
            "LAMBDA_L2": "LAMBDA_L2_entry",
            "NUM_TRIALS": "NUM_TRIALS_entry",
        }
        for key, attr in entry_mapping.items():
            if key in arg_name:
                setattr(self, attr, entry)
                break

        # write default value to config
        self.write_var_to_config(arg_name, var.get())

        # trace var change, write new var value to config
        var.trace("w", lambda *args: self.write_var_to_config(arg_name, var.get()))


    def write_var_to_config(self, arg_name, var_value):
        if not isinstance(arg_name, list):
            raise ValueError("arg_name must be list")

        try:
            if isinstance(arg_name, list):
                if len(arg_name) == 1:
                    a = arg_name[0]
                    self.config[a] = var_value
                if len(arg_name) == 2:
                    a, b = arg_name
                    self.config[a][b] = var_value
                if len(arg_name) == 3:
                    a, b, c = arg_name
                    self.config[a][b][c] = var_value

            update_config(self.config)
        except Exception as e:
            pass
            # print(e)

    def load_dataset(self):
        try:
            self.load_dataset_output = DatasetLoader().load_dataset()
            messagebox.showinfo("Success", f"Dataset ``{self.config['DATASET_NAME']}`` loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset ``{self.config['DATASET_NAME']}``: {e}")

    def train_model(self):
        if hasattr(self, 'load_dataset_output'):
            try:
                self.train_model_output = ModelTrainer(self.load_dataset_output).train_model()
                messagebox.showinfo("Success", f"Model ``{self.config['MODEL_NAME']}`` trained successfully!")

                # once the model is trained, we update the parms for model in APP
                self.trace_dropdown(["MODEL_NAME"], self.config['MODEL_NAME'])
            except Exception as e:
                messagebox.showerror("Error", f"Failed to train model ``{self.config['MODEL_NAME']}``: {e}")
        else:
            messagebox.showwarning("Warning", "Please load the dataset first!")


    def evaluate_model(self):
        if hasattr(self, 'load_dataset_output') and hasattr(self, 'train_model_output'):
            try:
                self.evaluate_model_output = ModelEvaluator(self.load_dataset_output, self.train_model_output).evaluate_model()
                messagebox.showinfo("Success", f"Model ``{self.config['MODEL_NAME']}`` evaluated successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to evaluate model: ``{self.config['MODEL_NAME']}``: {e}")
        else:
            messagebox.showwarning("Warning", "Please load the dataset and train the model first!")


if __name__ == "__main__":
    app = FinOLApp()
    app.run()