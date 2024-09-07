import time
import json
import tkinter as tk

from PIL import Image, ImageTk
from tkinter import messagebox, ttk, font
from finol.data_layer import DatasetLoader
from finol.optimization_layer import ModelTrainer
from finol.evaluation_layer import ModelEvaluator
from finol.utils import ROOT_PATH, load_config, update_config, detect_device

# from webbrowser import open as open_browser

# def format_string(input_string):
#     words = input_string.split("_")
#     formatted_words = [word.capitalize() for word in words]
#     formatted_string = " ".join(formatted_words)
#
#     return formatted_string
import sv_ttk
import sys


class RedirectOutput:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, string):
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)

    def flush(self):
        pass


class FinOLAPP:
    def __init__(self):
        self.config = load_config()
        # configure grid layout (4x4)
        self.root = tk.Tk()

        # default_font = font.nametofont("TkDefaultFont")
        # default_font.configure(weight="bold")
        # self.root.option_add("*Font", default_font)
        # print(font.families())

        # sv_ttk.set_theme("light")
        sv_ttk.set_theme("dark")

        # self.root.tk.call('source', 'forest-light.tcl')
        # ttk.Style().theme_use("vista")  # ('winnative', 'clam', 'alt', 'default', 'classic', 'vista', 'xpnative')

        # style = ttk.Style()
        # style.configure(
        #     'custom.TButton',
        #     font=("Arial", 12, "bold"))

        # self.root.state("zoomed")
        self.root.iconphoto(False, ImageTk.PhotoImage(Image.open(ROOT_PATH + "/APP/finol_logo_icon.png")))
        self.root.title("FinOL: Towards Open Benchmarking for Data-Driven Online Portfolio Selection")
        self.create_widgets()
        self.experiment_details = {}

    def apply_theme_to_titlebar(self, root):
        # version = sys.getwindowsversion()
        # import pywinstyles
        #
        # if version.major == 10 and version.build >= 22000:
        #     # Set the title bar color to the background color on Windows 11 for better appearance
        #     pywinstyles.change_header_color(root, "#1c1c1c" if sv_ttk.get_theme() == "dark" else "#fafafa")
        # elif version.major == 10:
        #     pywinstyles.apply_style(root, "dark" if sv_ttk.get_theme() == "dark" else "normal")
        #
        #     # A hacky way to update the title bar's color on Windows 10 (it doesn't update instantly like on Windows 11)
        #     root.wm_attributes("-alpha", 0.99)
        #     root.wm_attributes("-alpha", 1)
        pass

    # Example usage (replace `root` with the reference to your main/Toplevel window)

    def run(self):
        import platform

        def get_os():
            os_name = platform.system()
            if os_name == "Windows":
                self.apply_theme_to_titlebar(self.root)
                return "Windows"
            elif os_name == "Linux":
                return "Linux"
            elif os_name == "Darwin":
                return "macOS"
            else:
                return "Unknown"

        current_os = get_os()
        print(f"The current operating system is: {current_os}")
        self.root.mainloop()

    def restart_app(self):
        self.root.quit()

    def quit_app(self):
        import subprocess
        self.root.destroy()
        subprocess.call(["python", ROOT_PATH + "/APP/FinOLAPP.py"])

    def create_widgets(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill='both', expand=True)

        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side="left", fill="both", pady=10, expand=True)

        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side="left", fill="both", expand=True)

        # image_frame = ttk.Frame(main_frame)
        # image_frame.grid(row=0, column=0, sticky='ns')
        #
        # image_path = "../figure/finol_logo.png"
        # img = Image.open(image_path)
        # img = img.resize((100, 30))
        # img_tk = ImageTk.PhotoImage(img)
        #
        # image_label = ttk.Label(image_frame, image=img_tk)
        # image_label.image = img_tk
        # image_label.pack(pady=10)

        docs_frame = ttk.Frame(left_frame)
        docs_frame.pack(side='bottom', pady=10)

        docs_left = ttk.Label(docs_frame, text="View Docs ↗", foreground="azure", cursor="hand2")
        docs_left.pack(side='left', padx=8, pady=10)
        def open_docs_left(event):
            import webbrowser
            webbrowser.open_new("https://finol.readthedocs.io/en/latest/api/config.html")
        docs_left.bind("<Button-1>", open_docs_left)

        docs_middle = ttk.Label(docs_frame, text="Github ↗", foreground="azure", cursor="hand2")
        docs_middle.pack(side='left', padx=8, pady=10)
        def open_docs_middle(event):
            import webbrowser
            webbrowser.open_new("https://github.com/jiahaoli57/FinOL")
        docs_middle.bind("<Button-1>", open_docs_middle)

        # docs_right = ttk.Label(docs_frame, text="Contact us", foreground="blue", cursor="hand2")
        # docs_right.pack(side='left', padx=5)
        # def open_docs_right(event):
        #     import webbrowser
        #     webbrowser.open_new("https://finol.readthedocs.io/en/latest/api/config.html")
        # docs_right.bind("<Button-1>", open_docs_right)

        self.customize_dataset_button = ttk.Button(left_frame, text="Customize Dataset", width=15, command=self.customize_dataset)
        self.customize_dataset_button.pack(side='top', padx=20, pady=3)

        self.customize_model_button = ttk.Button(left_frame, text="Customize Model", width=15, command=self.customize_model)
        self.customize_model_button.pack(side='top', padx=20, pady=3)

        self.customize_criterion_button = ttk.Button(left_frame, text="Customize Criterion", width=15, command=self.customize_criterion)
        self.customize_criterion_button.pack(side='top', padx=20, pady=3)

        self.restart_button = ttk.Button(left_frame, text="Restart FinOL", width=15, command=self.restart_app)
        self.restart_button.pack(side='bottom', padx=20, pady=3)

        self.quit_button = ttk.Button(left_frame, text="Quit FinOL", width=15, command=self.quit_app)  # , style='custom.TButton'
        self.quit_button.pack(side='bottom', padx=20, pady=3)

        self.evaluate_button = ttk.Button(left_frame, text="Evaluate Model", width=15, command=self.evaluate_model)
        self.evaluate_button.pack(side='bottom', padx=20, pady=3)

        self.train_button = ttk.Button(left_frame, text="Train Model", width=15, command=self.train_model)
        self.train_button.pack(side='bottom', padx=20, pady=3)

        self.load_button = ttk.Button(left_frame, text="Load Dataset", width=15, command=self.load_dataset)
        self.load_button.pack(side='bottom', padx=20, pady=3)

        # tab_text_frame = ttk.Frame(main_frame)
        # tab_text_frame.grid(row=0, column=1, sticky='nsew')

        top_right_frame = ttk.Frame(right_frame)
        top_right_frame.pack(side="top", fill="both", expand=True)

        ##
        self.notebook = ttk.Notebook(top_right_frame)
        self.notebook.pack(side='left', fill='both', expand=True)
        #
        # self.general_config_frame = ttk.Frame(self.notebook)
        self.data_config_notebook = ttk.Frame(self.notebook)
        self.model_config_notebook = ttk.Frame(self.notebook)
        self.optimization_config_notebook = ttk.Frame(self.notebook)
        self.evaluation_config_notebook = ttk.Frame(self.notebook)
        #
        # self.notebook.add(self.general_config_frame, text="General Layer Configuration")
        self.notebook.add(self.data_config_notebook, text="Data Layer Configuration")
        self.notebook.add(self.model_config_notebook, text="Model Layer Configuration")
        self.notebook.add(self.optimization_config_notebook, text="Optimization Layer Configuration")
        self.notebook.add(self.evaluation_config_notebook, text="Evaluation Layer Configuration")
        #
        ###############################
        # General Layer Configuration #
        ###############################
        # self.general_config_frame = tk.LabelFrame(general_tab, text="General Configuration", font=("Helvetica", 10, "bold"))
        # self.general_config_frame.pack(padx=100, pady=1, fill="none")

        ############################
        # Data Layer Configuration #
        ############################
        self.data_config_frame = tk.LabelFrame(self.data_config_notebook, bd=0, padx=10, pady=10,)  # 透明外框
        self.data_config_frame.pack(padx=10, pady=1, fill="none")

        # DEVICE
        options = ["auto", "cpu", "cuda"]
        self.create_dropdown(self.data_config_frame, options, "Select Device:", 0, 0, options.index(self.config["DEVICE"]),
                             "StringVar", ["DEVICE"])
        # trace_dropdown with default value
        self.config = detect_device(self.config)
        update_config(self.config)

        # MANUAL_SEED
        self.create_entry(self.data_config_frame, "Set Seed:", 0, 2, self.config["MANUAL_SEED"], "IntVar", ["MANUAL_SEED"])

        # DATASET_NAME
        options = ["NYSE(O)", "NYSE(N)", "DJIA", "SP500", "TSE", "SSE", "HSI", "CMEG", "CRYPTO", "CustomDataset"]
        self.create_dropdown(self.data_config_frame, options, "Select Dataset:", 1, 0, options.index(self.config["DATASET_NAME"]), "StringVar", ["DATASET_NAME"])

        # SCALER
        options = ["None", "StandardScaler", "MinMaxScaler", "MaxAbsScaler", "RobustScaler", "WindowStandardScaler",
                   "WindowMinMaxScaler", "WindowMaxAbsScaler", "WindowRobustScaler"]
        self.create_dropdown(self.data_config_frame, options, "Select Scaler:", 1, 2, options.index(self.config["SCALER"]), "StringVar", ["SCALER"])

        # create_separator
        self.create_separator(frame=self.data_config_frame, row=2, text="Auto Feature Engineering")

        # INCLUDE_OHLCV_FEATURES
        self.create_checkbox(self.data_config_frame, "Include OHLCV Features", 3, 0, self.config["FEATURE_ENGINEERING_CONFIG"]["INCLUDE_OHLCV_FEATURES"], ["FEATURE_ENGINEERING_CONFIG", "INCLUDE_OHLCV_FEATURES"])

        # INCLUDE_OVERLAP_FEATURES
        self.create_checkbox(self.data_config_frame, "Include Overlap Features", 3, 1, self.config["FEATURE_ENGINEERING_CONFIG"]["INCLUDE_OVERLAP_FEATURES"], ["FEATURE_ENGINEERING_CONFIG", "INCLUDE_OVERLAP_FEATURES"])

        # INCLUDE_MOMENTUM_FEATURES
        self.create_checkbox(self.data_config_frame, "Include Momentum Features", 3, 2, self.config["FEATURE_ENGINEERING_CONFIG"]["INCLUDE_MOMENTUM_FEATURES"], ["FEATURE_ENGINEERING_CONFIG", "INCLUDE_MOMENTUM_FEATURES"])

        # INCLUDE_VOLUME_FEATURES
        self.create_checkbox(self.data_config_frame, "Include Volume Features", 3, 3, self.config["FEATURE_ENGINEERING_CONFIG"]["INCLUDE_VOLUME_FEATURES"], ["FEATURE_ENGINEERING_CONFIG", "INCLUDE_VOLUME_FEATURES"])

        # INCLUDE_CYCLE_FEATURES
        self.create_checkbox(self.data_config_frame, "Include Cycle Features", 4, 0, self.config["FEATURE_ENGINEERING_CONFIG"]["INCLUDE_CYCLE_FEATURES"], ["FEATURE_ENGINEERING_CONFIG", "INCLUDE_CYCLE_FEATURES"])

        #  INCLUDE_PRICE_FEATURES
        self.create_checkbox(self.data_config_frame, "Include Price Features", 4, 1, self.config["FEATURE_ENGINEERING_CONFIG"]["INCLUDE_PRICE_FEATURES"], ["FEATURE_ENGINEERING_CONFIG", "INCLUDE_PRICE_FEATURES"])

        # INCLUDE_VOLATILITY_FEATURES
        self.create_checkbox(self.data_config_frame, "Include Volatility Features", 4, 2, self.config["FEATURE_ENGINEERING_CONFIG"]["INCLUDE_VOLATILITY_FEATURES"], ["FEATURE_ENGINEERING_CONFIG", "INCLUDE_VOLATILITY_FEATURES"])

        #  INCLUDE_PATTERN_FEATURES
        self.create_checkbox(self.data_config_frame, "Include Pattern Features", 4, 3, self.config["FEATURE_ENGINEERING_CONFIG"]["INCLUDE_PATTERN_FEATURES"], ["FEATURE_ENGINEERING_CONFIG", "INCLUDE_PATTERN_FEATURES"])

        # create_separator
        self.create_separator(frame=self.data_config_frame, row=5, text="Data Augmentation")

        # INCLUDE_WINDOW_DATA
        self.create_checkbox(self.data_config_frame, "Include Window Data", 6, 0, self.config["DATA_AUGMENTATION_CONFIG"]["WINDOW_DATA"]["INCLUDE_WINDOW_DATA"], ["DATA_AUGMENTATION_CONFIG", "WINDOW_DATA", "INCLUDE_WINDOW_DATA"])

        # WINDOW_SIZE
        self.create_entry(self.data_config_frame, "Set Window Size:", 6, 1, self.config["DATA_AUGMENTATION_CONFIG"]["WINDOW_DATA"]["WINDOW_SIZE"], "IntVar", ["DATA_AUGMENTATION_CONFIG", "WINDOW_DATA", "WINDOW_SIZE"])
        # trace_checkbox with default value
        self.trace_checkbox(["DATA_AUGMENTATION_CONFIG", "WINDOW_DATA", "INCLUDE_WINDOW_DATA"], self.config["DATA_AUGMENTATION_CONFIG"]["WINDOW_DATA"]["INCLUDE_WINDOW_DATA"])

        # create_separator
        self.create_separator(frame=self.data_config_frame, row=7)

        # LOAD_LOCAL_DATALOADER
        self.create_checkbox(self.data_config_frame, "Load Local Dataloader", 8, 0, self.config["LOAD_LOCAL_DATALOADER"], ["LOAD_LOCAL_DATALOADER"])
        # trace_checkbox with default value
        self.trace_checkbox(["LOAD_LOCAL_DATALOADER"], self.config["LOAD_LOCAL_DATALOADER"])

        # DOWNLOAD_DATA
        self.create_checkbox(self.data_config_frame, "Download FinOL Data", 8, 1, self.config["DOWNLOAD_DATA"], ["DOWNLOAD_DATA"])

        # CHECK_UPDATE
        self.create_checkbox(self.data_config_frame, "Check for Updates", 8, 2, self.config["CHECK_UPDATE"], ["CHECK_UPDATE"])

        #############################
        # Model Layer Configuration #
        #############################
        self.model_config_frame = tk.LabelFrame(self.model_config_notebook, bd=0, padx=10, pady=10,)
        self.model_config_frame.pack(padx=10, pady=1, fill="none")

        # ttk.Label(self.model_config_frame, text=" "*150).grid(row=0, column=0, columnspan=4, padx=10, pady=0)
        # ttk.Label(self.model_config_frame, text=" "*150).grid(row=100, column=0, columnspan=4, padx=10, pady=0)

        # MODEL_NAME
        options = ["AlphaPortfolio", "CNN", "DNN", "LSRE-CAAN", "LSTM", "RNN", "Transformer", "CustomModel"]
        self.create_dropdown(self.model_config_frame, options, "Select Model:", 0, 1, options.index(self.config["MODEL_NAME"]), "StringVar", ["MODEL_NAME"])
        # create_separator
        self.create_separator(frame=self.model_config_frame, row=1, text="Model Parameters")
        # trace_dropdown with default value
        self.trace_dropdown(["MODEL_NAME"], self.config["MODEL_NAME"])


        ####################################
        # Optimization Layer Configuration #
        ####################################
        self.optimization_config_frame = tk.LabelFrame(self.optimization_config_notebook, bd=0, padx=10, pady=10,)
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
        options = ["LogWealth", "LogWealthL2Diversification", "LogWealthL2Concentration", "L2Diversification", "L2Concentration", "SharpeRatio", "Volatility", "CustomCriterion"]
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


        ##################################
        # Evaluation Layer Configuration #
        ##################################
        self.evaluation_config_frame = tk.LabelFrame(self.evaluation_config_notebook, bd=0, padx=10, pady=10,)
        self.evaluation_config_frame.pack(padx=10, pady=1, fill="none")

        # PLOT_LANGUAGE
        options = ["en", "zh_CN", "zh_TW"]
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

        #############################################

        # self.result_frame = tk.LabelFrame(self.root, text="Results")
        # self.result_frame.pack(padx=10, pady=1, fill="both", expand=True)

        bottom_right_frame = ttk.Frame(right_frame, height=30, borderwidth=0)
        bottom_right_frame.pack(side="bottom", fill="both", expand=True)

        # 放置文本框
        # self.text = tk.Text(bottom_right_frame, width=130, height=15, font="TkDefaultFont", )
        # self.text_widget = tk.Text(bottom_right_frame, wrap=tk.WORD, width=130, height=20, font="TkDefaultFont", fg="blue", bg="lightyellow", bd=2, padx=10, pady=10, borderwidth=0)
        self.text_widget = tk.Text(bottom_right_frame, wrap=tk.WORD, width=160, height=30,  bg="gray16", bd=2, padx=10, pady=10, borderwidth=0)
        self.text_widget.pack(side=tk.LEFT, padx=10, pady=10, fill='both', expand=True)

        # 添加滚动条
        scrollbar = ttk.Scrollbar(bottom_right_frame)  # 将滚动条放在 bottom_right_frame 上
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_widget.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.text_widget.yview)

        # 保存原始的 sys.stdout
        original_stdout = sys.stdout
        # 在这里将 sys.stdout 设置为 RedirectOutput 类的实例 self.text
        sys.stdout = RedirectOutput(self.text_widget)
        from finol.APP import display_info
        # 在需要取消重定向时，将 sys.stdout 重定向回原始的 sys.stdout
        sys.stdout = original_stdout

       #  self.text_widget.insert(
       #      tk.END,
       #      tabulate(profitability_table, headers="firstrow", tablefmt="psql", numalign="left")
       # )
        # Redirect stdout and stderr
        # sys.stdout = RedirectOutput(self.text)
        # sys.stderr = RedirectOutput(self.text)  #

        # import logging
        # # Setup logging
        # logging.basicConfig(level=logging.INFO)
        # logger = logging.getLogger()
        #
        # # Redirect logging to Text widget
        # log_handler = logging.StreamHandler(RedirectOutput(sROOT_PATHelf.text))
        # logger.addHandler(log_handler)

        # self.to_all_frame = ttk.Frame(top_right_frame)
        # self.to_all_frame.pack(side="left", fill="both", expand=True)
        #
        # to_all = tk.Text(self.to_all_frame)
        # to_all.pack(side="left", fill="both", expand=True)

    def customize_dataset(self):
        self.text_widget.insert(
            tk.END,
            "\n--------------------------------\n"
            " To customize your own dataset: \n"
            "--------------------------------\n\n"
            f"1. Navigate to the \"{ROOT_PATH}\\data\\datasets\CustomDataset\" directory.\n"
            f"2. Create .xlsx files for different assets in the following format:\n"
            f" +------------+----------+----------+----------+----------+---------+\n"
            " | DATE       | OPEN     |  HIGH    | LOW      | CLOSE    | VOLUME  |\n"
            " |------------+----------+----------+----------+----------+---------|\n"
            " | 2017-11-09 | 0.025160 |	0.035060 | 0.025006	| 0.032053 | 1871620 |\n"
            " | 2017-11-10 | 0.032219 |	0.033348 | 0.026450	| 0.027119 | 6766780 |\n"
            " | 2017-11-11 | 0.026891 |	0.029658 | 0.025684	| 0.027437 | 5532220 |\n"
            " | 2017-11-12 | 0.027480 |	0.027952 | 0.022591	| 0.023977 | 7280250 |\n"
            " | 2017-11-13 | 0.024364 |	0.026300 | 0.023495	| 0.025807 | 4419440 |\n"
            " | 2017-11-14 | 0.025797 |	0.026788 | 0.025342	| 0.026230 | 3033290 |\n"
            " | 2017-11-15 | 0.026116 |	0.027773 | 0.025261	| 0.026445 | 6858800 |\n"
            " | ......     | ......   |	......   | ......  	| ......   | ......  |\n"
            " | 2024-02-29 | 0.630859 |	0.705280 | 0.625720	| 0.655646 | 1639531 |\n"
            " | 2024-03-01 | 0.655440 |	0.719080 | 0.654592	| 0.719080 | 9353798 |\n"
            f" +-----------+-----------+----------+----------+----------+---------+\n"
            f"3. For each asset, ensure that the data are correctly formatted and there are no missing values.\n"
            f"4. Define the configuration for your custom dataset in the \"{ROOT_PATH}\\config.json\" file, under the \"[\"DATASET_SPLIT_CONFIG\"][\"CustomModel\"]\", \"[\"BATCH_SIZE\"][\"CustomModel\"]\", and \"[\"NUM_DAYS_PER_YEAR\"][\"CustomModel\"]\" sections.\n\n"
            f"NOTE: Instead of customizing the dataset yourself, we recommend that you raise an issue or contact us by "
            f"email so we can evaluate and potentially include your dataset in the FinOL project. This ensures the "
            f"baseline results are supported.\n")

        self.text_widget.yview_moveto(1.0)  # Scroll to the bottom

    def customize_model(self):
        self.text_widget.insert(
            tk.END,
            "\n------------------------------\n"
            " To customize your own model: \n"
            "------------------------------\n\n"
            f"1. Navigate to the \"{ROOT_PATH}\\model_layer\\CustomModel.py\" file in the FinOL codebase.\n"
            f"2. Customize your own model by extending \"CustomModel\" class. This is where you will implement the logic for your custom data-driven OLPS model.\n"
            f"3. Define the necessary hyper-parameters in the \"{ROOT_PATH}\\config.json\" file, under the \"[\"MODEL_PARAMS\"][\"CustomModel\"]\" section.\n"
            f"4. (Optional) If you want FinOL to automatically tune the hyper-parameters of your custom model, "
            f"specify the range of different parameters in the \"MODEL_PARAMS_SPACE[\"CustomModel\"]\" section of the \"{ROOT_PATH}\\config.json\" file.\n")

        self.text_widget.yview_moveto(1.0)  # Scroll to the bottom

    def customize_criterion(self):
        self.text_widget.insert(
            tk.END,
            "\n---------------------------------\n"
            " To customize your own criterion: \n"
            "---------------------------------\n\n"
            f"1. Navigate to \"{ROOT_PATH}\\optimization_layer\\criterion_selector.py\".\n"
            f"2. Locate the \"CustomCriterion\" class and define your own custom loss function by overriding the \"compute_custom_criterion_loss\" method.\n"
            f"3. This allows you totailor the optimization objective to your specific requirements.\n")

        self.text_widget.yview_moveto(1.0)  # Scroll to the bottom

    def create_checkbox(self, frame, text, row, column, default_value, arg_name):
        pady = 3
        var = tk.BooleanVar(value=default_value)
        checkbox = ttk.Checkbutton(frame, text=text, variable=var)
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

            # double check
            if self.config["PRUNER_NAME"] != "PatientPruner":
                self.WRAPPED_PRUNER_NAME_dropdown.config(state="disabled")

        if "INCLUDE_WINDOW_DATA" in arg_name:
            widgets = [self.WINDOW_SIZE_entry,]
            state = "normal" if var_value else "disabled"
            for widget in widgets:
                widget.config(state=state)

    def create_dropdown(self, frame, options, text, row, column, default_value, value_type, arg_name):
        pady = 3
        ttk.Label(frame, text=text).grid(row=row, column=column, padx=10, pady=pady)
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
            state = "normal" if var_value == "LogWealthL2Diversification" or var_value == "LogWealthL2Concentration" else "disabled"
            for widget in widgets:
                widget.config(state=state)

        if "MODEL_NAME" in arg_name:
            # reload the self.config as the config might change when running the optuna_optimizer.py
            self.config = load_config()
            #
            MODEL_NAME = var_value
            for widget in self.model_config_frame.winfo_children():
                # print(str(widget))
                if str(widget) in [".!frame.!frame2.!frame.!notebook.!frame2.!labelframe.!label",
                                   ".!frame.!frame2.!frame.!notebook.!frame2.!labelframe.!combobox",
                                   ".!frame.!frame2.!frame.!notebook.!frame2.!labelframe.!separator",
                                   ".!frame.!frame2.!frame.!notebook.!frame2.!labelframe.!label2"]:
                    pass
                else:
                    widget.destroy()

            # write default value to config
            self.write_var_to_config(["MODEL_NAME"], MODEL_NAME)

            # create_separator
            # self.create_separator(frame=self.model_config_frame, row=2, text="Model Parameters")
            # self.create_separator(frame=self.model_config_frame, row=1, text="Model Parameters")

            default_model_parms = self.config["MODEL_PARAMS"][MODEL_NAME]
            if MODEL_NAME == "AlphaPortfolio":
                self.create_entry(self.model_config_frame, "Number of Layers:", row=3, column=0, default_value=default_model_parms["NUM_LAYERS"],
                                  value_type="IntVar", arg_name=["MODEL_PARAMS", MODEL_NAME, "NUM_LAYERS"])
                self.create_entry(self.model_config_frame, "Dimension of Embedding:", row=3, column=2, default_value=default_model_parms["DIM_EMBEDDING"],
                                  value_type="IntVar", arg_name=["MODEL_PARAMS", MODEL_NAME, "DIM_EMBEDDING"])
                self.create_entry(self.model_config_frame, "Dimension of Feedforward:", row=4, column=0, default_value=default_model_parms["DIM_FEEDFORWARD"],
                                  value_type="IntVar", arg_name=["MODEL_PARAMS", MODEL_NAME, "DIM_FEEDFORWARD"])
                self.create_entry(self.model_config_frame, "Number of Heads:", row=4, column=2, default_value=default_model_parms["NUM_HEADS"],
                                  value_type="IntVar", arg_name=["MODEL_PARAMS", MODEL_NAME, "NUM_HEADS"])
                self.create_entry(self.model_config_frame, "Dropout Rate:", row=5, column=0, default_value=default_model_parms["DROPOUT"],
                                  value_type="DoubleVar", arg_name=["MODEL_PARAMS", MODEL_NAME, "DROPOUT"])

            elif MODEL_NAME == "CNN":
                self.create_entry(self.model_config_frame, "Out Channels:", row=3, column=0, default_value=default_model_parms["OUT_CHANNELS"],
                                  value_type="IntVar", arg_name=["MODEL_PARAMS", MODEL_NAME, "OUT_CHANNELS"])
                self.create_entry(self.model_config_frame, "Kernel Size:", row=3, column=2, default_value=default_model_parms["KERNEL_SIZE"],
                                  value_type="IntVar", arg_name=["MODEL_PARAMS", MODEL_NAME, "KERNEL_SIZE"])
                self.create_entry(self.model_config_frame, "Stride:", row=4, column=0, default_value=default_model_parms["STRIDE"],
                                  value_type="IntVar", arg_name=["MODEL_PARAMS", MODEL_NAME, "STRIDE"])
                self.create_entry(self.model_config_frame, "Hidden Size:", row=4, column=2, default_value=default_model_parms["HIDDEN_SIZE"],
                                  value_type="IntVar", arg_name=["MODEL_PARAMS", MODEL_NAME, "HIDDEN_SIZE"])
                self.create_entry(self.model_config_frame, "Dropout Rate:", row=5, column=0, default_value=default_model_parms["DROPOUT"],
                                  value_type="DoubleVar", arg_name=["MODEL_PARAMS", MODEL_NAME, "DROPOUT"])

            elif MODEL_NAME == "DNN":
                self.create_entry(self.model_config_frame, "Number of Layers:", row=3, column=0, default_value=default_model_parms["NUM_LAYERS"],
                                  value_type="IntVar", arg_name=["MODEL_PARAMS", MODEL_NAME, "NUM_LAYERS"])
                self.create_entry(self.model_config_frame, "Hidden Size:", row=3, column=2, default_value=default_model_parms["HIDDEN_SIZE"],
                                  value_type="IntVar", arg_name=["MODEL_PARAMS", MODEL_NAME, "HIDDEN_SIZE"])
                self.create_entry(self.model_config_frame, "Dropout Rate:", row=4, column=0, default_value=default_model_parms["DROPOUT"],
                                  value_type="DoubleVar", arg_name=["MODEL_PARAMS", MODEL_NAME, "DROPOUT"])

            elif MODEL_NAME == "LSRE-CAAN":
                self.create_entry(self.model_config_frame, "Number of Layers:", row=3, column=0, default_value=default_model_parms["NUM_LAYERS"],
                                  value_type="IntVar", arg_name=["MODEL_PARAMS", MODEL_NAME, "NUM_LAYERS"])
                self.create_entry(self.model_config_frame, "Number of Latents:", row=3, column=2, default_value=default_model_parms["NUM_LATENTS"],
                                  value_type="IntVar", arg_name=["MODEL_PARAMS", MODEL_NAME, "NUM_LATENTS"])
                self.create_entry(self.model_config_frame, "Dimension of Latent:", row=4, column=0, default_value=default_model_parms["LATENT_DIM"],
                                  value_type="IntVar", arg_name=["MODEL_PARAMS", MODEL_NAME, "LATENT_DIM"])
                self.create_entry(self.model_config_frame, "Number of Cross Heads:", row=4, column=2, default_value=default_model_parms["CROSS_HEADS"],
                                  value_type="IntVar", arg_name=["MODEL_PARAMS", MODEL_NAME, "CROSS_HEADS"])
                self.create_entry(self.model_config_frame, "Number of Latent Heads:", row=5, column=0, default_value=default_model_parms["LATENT_HEADS"],
                                  value_type="IntVar", arg_name=["MODEL_PARAMS", MODEL_NAME, "LATENT_HEADS"])
                self.create_entry(self.model_config_frame, "Dimensions per Cross Head:", row=5, column=2, default_value=default_model_parms["CROSS_DIM_HEAD"],
                                  value_type="IntVar", arg_name=["MODEL_PARAMS", MODEL_NAME, "CROSS_DIM_HEAD"])
                self.create_entry(self.model_config_frame, "Dimensions per Latent Head:", row=6, column=0, default_value=default_model_parms["LATENT_DIM_HEAD"],
                                  value_type="IntVar", arg_name=["MODEL_PARAMS", MODEL_NAME, "LATENT_DIM_HEAD"])
                self.create_entry(self.model_config_frame, "Dropout Rate:", row=6, column=2, default_value=default_model_parms["DROPOUT"],
                                  value_type="DoubleVar", arg_name=["MODEL_PARAMS", MODEL_NAME, "DROPOUT"])

            elif MODEL_NAME == "LSTM":
                self.create_entry(self.model_config_frame, "Number of Layers:", row=3, column=0, default_value=default_model_parms["NUM_LAYERS"],
                                  value_type="IntVar", arg_name=["MODEL_PARAMS", MODEL_NAME, "NUM_LAYERS"])
                self.create_entry(self.model_config_frame, "Hidden Size:", row=3, column=2, default_value=default_model_parms["HIDDEN_SIZE"],
                                  value_type="IntVar", arg_name=["MODEL_PARAMS", MODEL_NAME, "HIDDEN_SIZE"])
                self.create_entry(self.model_config_frame, "Dropout Rate:", row=4, column=0, default_value=default_model_parms["DROPOUT"],
                                  value_type="DoubleVar", arg_name=["MODEL_PARAMS", MODEL_NAME, "DROPOUT"])

            elif MODEL_NAME == "RNN":
                self.create_entry(self.model_config_frame, "Number of Layers:", row=3, column=0, default_value=default_model_parms["NUM_LAYERS"],
                                  value_type="IntVar", arg_name=["MODEL_PARAMS", MODEL_NAME, "NUM_LAYERS"])
                self.create_entry(self.model_config_frame, "Hidden Size:", row=3, column=2, default_value=default_model_parms["HIDDEN_SIZE"],
                                  value_type="IntVar", arg_name=["MODEL_PARAMS", MODEL_NAME, "HIDDEN_SIZE"])
                self.create_entry(self.model_config_frame, "Dropout Rate:", row=4, column=0, default_value=default_model_parms["DROPOUT"],
                                  value_type="DoubleVar", arg_name=["MODEL_PARAMS", MODEL_NAME, "DROPOUT"])

            elif MODEL_NAME == "Transformer":
                self.create_entry(self.model_config_frame, "Number of Layers:", row=3, column=0, default_value=default_model_parms["NUM_LAYERS"],
                                  value_type="IntVar", arg_name=["MODEL_PARAMS", MODEL_NAME, "NUM_LAYERS"])
                self.create_entry(self.model_config_frame, "Dimension of Embedding:", row=3, column=2, default_value=default_model_parms["DIM_EMBEDDING"],
                                  value_type="IntVar", arg_name=["MODEL_PARAMS", MODEL_NAME, "DIM_EMBEDDING"])
                self.create_entry(self.model_config_frame, "Dimension of Feedforward:", row=4, column=0, default_value=default_model_parms["DIM_FEEDFORWARD"],
                                  value_type="IntVar", arg_name=["MODEL_PARAMS", MODEL_NAME, "DIM_FEEDFORWARD"])
                self.create_entry(self.model_config_frame, "Number of Heads:", row=4, column=2, default_value=default_model_parms["NUM_HEADS"],
                                  value_type="IntVar", arg_name=["MODEL_PARAMS", MODEL_NAME, "NUM_HEADS"])
                self.create_entry(self.model_config_frame, "Dropout Rate:", row=5, column=0, default_value=default_model_parms["DROPOUT"],
                                  value_type="DoubleVar", arg_name=["MODEL_PARAMS", MODEL_NAME, "DROPOUT"])

            elif MODEL_NAME == "CustomModel":
                self.create_entry(self.model_config_frame, "Setting of Parameter 1:", row=3, column=0, default_value=default_model_parms["PARAMETER1"],
                                  value_type="IntVar", arg_name=["MODEL_PARAMS", MODEL_NAME, "PARAMETER1"])
                self.create_entry(self.model_config_frame, "Setting of Parameter 2:", row=3, column=2, default_value=default_model_parms["PARAMETER2"],
                                  value_type="IntVar", arg_name=["MODEL_PARAMS", MODEL_NAME, "PARAMETER2"])

            # # trace var change, write new var value to config
            # self.MODEL_NAME_var.trace("w", lambda *args: self.write_var_to_config(["MODEL_NAME"], MODEL_NAME))

        if "PRUNER_NAME" in arg_name:
            widgets = [self.WRAPPED_PRUNER_NAME_dropdown,]
            state = "normal" if var_value == "PatientPruner" else "disabled"
            for widget in widgets:
                widget.config(state=state)

    def create_separator(self, frame, row, text=None):
        separator = ttk.Separator(frame, orient="horizontal")
        separator.grid(row=row, column=0, columnspan=6, padx=10, pady=1, sticky="ew")
        # separator_label = ttk.Label(frame)  # , text="-"*210
        # separator_label.grid(row=row, column=0, columnspan=6, padx=10, pady=1)
        if text != None:
            separator_label = ttk.Label(frame, text=text)
            separator_label.grid(row=row, column=0, columnspan=6, padx=10, pady=1)

    def create_entry(self, frame, text, row, column, default_value, value_type, arg_name):
        pady = 3
        ttk.Label(frame, text=text).grid(row=row, column=column, padx=10, pady=pady)
        var = getattr(tk, value_type)()
        entry = ttk.Entry(frame, textvariable=var)
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
            messagebox.showinfo("Success", f"Dataset [{self.config['DATASET_NAME']}] loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset ``{self.config['DATASET_NAME']}``: {e}")

    def train_model(self):
        if hasattr(self, 'load_dataset_output'):
            try:
                self.train_model_output = ModelTrainer(self.load_dataset_output).train_model()
                # once the model is trained, we update the parms for model in APP
                self.trace_dropdown(["MODEL_NAME"], self.config['MODEL_NAME'])

                messagebox.showinfo("Success", f"Model [{self.config['MODEL_NAME']}] trained successfully!")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to train model ``{self.config['MODEL_NAME']}``: {e}")
        else:
            messagebox.showwarning("Warning", "Please load the dataset first!")


    def evaluate_model(self):
        if hasattr(self, 'load_dataset_output') and hasattr(self, 'train_model_output'):
            try:
                self.evaluate_model_output = ModelEvaluator(self.load_dataset_output, self.train_model_output).evaluate_model()
                messagebox.showinfo("Success", f"Model [{self.config['MODEL_NAME']}] evaluated successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to evaluate model: ``{self.config['MODEL_NAME']}``: {e}")
        else:
            messagebox.showwarning("Warning", "Please load the dataset and train the model first!")


if __name__ == "__main__":
    app = FinOLAPP()
    app.run()