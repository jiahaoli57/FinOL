import time
import torch
import optuna
import optuna.visualization.matplotlib as mpl
import matplotlib.pyplot as plt

from tqdm import tqdm
from tabulate import tabulate
from finol.model_layer.model_instantiator import ModelInstantiator
from finol.optimization_layer.criterion_selector import CriterionSelector
from finol.optimization_layer.optimizer_selector import OptimizerSelector
from finol.utils import load_config, update_config, portfolio_selection, actual_portfolio_selection, set_seed, add_prefix


class ParametersTuner:
    """
    Class to tune model hyper-parameters using auto ML library.

    :param load_dataset_output:
    Example:
        .. code:: python
        >>> # Load dataset
        >>> from finol.data_layer.dataset_loader import DatasetLoader
        >>> load_dataset_output = DatasetLoader().load_dataset()
        >>>
        >>> # Tune model
        >>> ParametersTuner(load_dataset_output=load_dataset_output).tune_parameters()

    """
    def __init__(self, load_dataset_output) -> None:
        super().__init__()
        self.config = load_config()
        self.load_dataset_output = load_dataset_output
        self.train_loader = load_dataset_output["train_loader"]
        self.val_loader = load_dataset_output["val_loader"]
        self.test_loader = load_dataset_output["test_loader_"]
        self.logdir = load_dataset_output["logdir"]

    def sample_params(self, trial: optuna.Trial) -> None:
        """
        Samples model hyper-parameters for optimization.

        :param trial: Optuna Trial object to sample the parameters.
        :return: None
        """
        model_sampled_params = self.config["MODEL_PARAMS_SPACE"][self.config["MODEL_NAME"]]
        self.sampled_params = {}

        for param_name, param_space in model_sampled_params.items():
            param_type = param_space["type"]
            param_range = param_space["range"]
            param_step = param_space["step"]

            if param_type == "int":
                self.sampled_params[param_name] = trial.suggest_int(param_name, *param_range, step=param_step)
            elif param_type == "float":
                self.sampled_params[param_name] = trial.suggest_float(param_name, *param_range, step=param_step)
            else:
                raise ValueError(f"Invalid parameter type: {param_type}")

    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization. Trains the model using the sampled parameters and returns the validation loss.

        :param trial: Optuna Trial object.
        :return: Validation loss.
        """
        self.sample_params(trial)
        set_seed(seed=self.config["MANUAL_SEED"])

        model = ModelInstantiator(self.load_dataset_output).instantiate_model(self.sampled_params)
        optimizer = OptimizerSelector(model).select_optimizer()
        criterion = CriterionSelector()

        train_loss_list = []
        val_loss_list = []
        test_loss_list = []
        best_val_loss = float("inf")
        value = float("inf")

        for e in tqdm(range(self.config["NUM_EPOCHES"]), desc="Training"):
            model.train()
            train_loss = 0
            for i, data in enumerate(self.train_loader, 1):
                x_data, label = data
                final_scores = model(x_data.float())
                portfolio = portfolio_selection(final_scores)
                loss = criterion(portfolio, label.float())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(self.train_loader)
            train_loss_list.append(train_loss)

            if (e + 1) % self.config["SAVE_EVERY"] == 0:
                with torch.no_grad():
                    model.eval()
                    val_loss = 0
                    for i, data in enumerate(self.val_loader, 1):
                        val_data, label = data
                        final_scores = model(val_data.float())
                        portfolio = portfolio_selection(final_scores)
                        loss = criterion(portfolio, label.float())
                        val_loss += loss.item()

                    val_loss /= len(self.val_loader)
                    val_loss_list.append(val_loss)
                    # value = sum(val_loss_list) / len(val_loss_list)
                    value = min(val_loss_list)  # use this
                    # value = val_loss  # real time val loss

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss

            # Report intermediate objective value.
            trial.report(value, e)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.TrialPruned()
        return value

    def select_sampler(self) -> object:
        """
        Selects and initializes an Optuna sampler based on the configuration.

        :return: Initialized Optuna sampler object.
        """
        sampler_name = self.config["SAMPLER_NAME"]
        sampler_kwargs = {"seed": self.config["MANUAL_SEED"]}

        if sampler_name == "GridSampler":
            model_sampled_params = self.config["MODEL_PARAMS_SPACE"][self.config["MODEL_NAME"]]
            search_space = {}
            for param_name, param_space in model_sampled_params.items():
                param_type = param_space["type"]
                param_range = param_space["range"]
                if param_type == "int" or param_type == "float":
                    search_space[param_name] = param_range
            sampler_kwargs = {"search_space": search_space}

        return getattr(optuna.samplers, sampler_name)(**sampler_kwargs)

    def select_pruner(self) -> object:
        """
        Selects and initializes an Optuna pruner based on the configuration.

        :return: Initialized Optuna pruner object.
        """
        pruner_name = self.config["PRUNER_NAME"]
        pruner_kwargs = {}

        if pruner_name == "PatientPruner":
            pruner_kwargs["wrapped_pruner"] = getattr(optuna.pruners, self.config["WRAPPED_PRUNER_NAME"])()
            pruner_kwargs["patience"] = 1

        return getattr(optuna.pruners, pruner_name)(**pruner_kwargs)

    def tune_parameters(self) -> None:
        """
        Tune model hyper-parameters.

        This method creates an Optuna study object, optimizes the objective function, visualizes the results, and prints out the optimization results.

        :return: None
        """
        # Creating Optuna object and defining its parameters
        self.study = optuna.create_study(
            direction="minimize",
            sampler=self.select_sampler(),
            pruner=self.select_pruner(),
            storage="sqlite:///" + self.logdir + "/" + add_prefix(".db")
        )
        self.study.optimize(self.objective, n_trials=self.config["NUM_TRIALS"])

        # To further visualize the results, you can upload the generated {MODEL_NAME}.db file to the Optuna Dashboard:
        # https://optuna.github.io/optuna-dashboard/
        # optuna.visualization.plot_optimization_history(self.study).show()
        plots = [
            mpl.plot_contour,
            mpl.plot_edf,
            # mpl.plot_hypervolume_history,
            mpl.plot_intermediate_values,
            mpl.plot_optimization_history,
            mpl.plot_parallel_coordinate,
            mpl.plot_param_importances,
            # mpl.plot_pareto_front,
            mpl.plot_rank,
            mpl.plot_slice,
            # mpl.plot_terminator_improvement,  # time consuming
            mpl.plot_timeline,
        ]
        for plot_func in plots:
            plt.figure()
            plot_func(self.study)
            plt.savefig(self.logdir + "/" + add_prefix(plot_func.__name__) + ".pdf", format="pdf", dpi=300, bbox_inches="tight")
            plt.show()
            # plt.tight_layout()

        # Showing optimization results
        tabulate_data = [
            ["Sampler", self.config["SAMPLER_NAME"]],
            ["Pruner", self.config["PRUNER_NAME"]],
            ["Number of finished trials", len(self.study.trials)],
            ["Best trial parameters", self.study.best_trial.params],
            ["Best score", self.study.best_value]
        ]
        print(tabulate(tabulate_data, headers=["Auto Hyper-parameters Tuning", "INFO"], tablefmt="psql"))  # , headers=["Metric", "Value"]

        # Write config
        self.config["MODEL_PARAMS"][self.config["MODEL_NAME"]] = self.study.best_trial.params
        update_config(self.config)


if __name__ == "__main__":
    from finol.data_layer.dataset_loader import DatasetLoader
    load_dataset_output = DatasetLoader().load_dataset()
    ParametersTuner(load_dataset_output=load_dataset_output).tune_parameters()
