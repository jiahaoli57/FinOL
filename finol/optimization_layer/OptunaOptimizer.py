import torch
import optuna
import matplotlib.pyplot as plt

from tqdm import tqdm
from finol.model_layer.ModelSelector import ModelSelector
from finol.optimization_layer.CriterionSelector import CriterionSelector
from finol.optimization_layer.OptimizerSelector import OptimizerSelector
from finol.utils import load_config, update_config, portfolio_selection, set_seed


class OptunaOptimizer:
    def __init__(self, load_dataset_output):
        super().__init__()
        self.config = load_config()
        self.load_dataset_output = load_dataset_output
        self.train_loader = load_dataset_output["train_loader"]
        self.val_loader = load_dataset_output["val_loader"]
        self.logdir = load_dataset_output["logdir"]

    def sample_params(self, trial: optuna.Trial):
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

    def objective(self, trial: optuna.Trial):
        self.sample_params(trial)
        set_seed(seed=self.config["MANUAL_SEED"])

        model = ModelSelector(self.load_dataset_output).select_model(self.sampled_params)
        optimizer = OptimizerSelector(model).select_optimizer()
        criterion = CriterionSelector()

        train_loss_list = []
        val_loss_list = []
        best_val_loss = float("inf")

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
                    value = min(val_loss_list)
                    # value = val_loss  # real time val loss

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss

            # Report intermediate objective value.
            trial.report(value, e)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.TrialPruned()

        return value

    def train_via_optuna(self):
        # Creating Optuna object and defining its parameters
        self.study = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.TPESampler(seed=self.config["MANUAL_SEED"]),
            pruner=optuna.pruners.MedianPruner(),
            storage="sqlite:///" + self.logdir + "/" + self.config["MODEL_NAME"] + "_" + self.config["DATASET_NAME"] + "_OPTUNA_" + self.config["MODEL_NAME"] + ".db"
        )
        print(f"Sampler is {self.study.sampler.__class__.__name__}")
        self.study.optimize(self.objective, n_trials=self.config["NUM_TRIALS"])

        # To further visualize the results, you can upload the generated {MODEL_NAME}.db file to the Optuna Dashboard:
        # https://optuna.github.io/optuna-dashboard/
        optuna.visualization.plot_intermediate_values(self.study).show()
        optuna.visualization.plot_optimization_history(self.study).show()

        # Showing optimization results
        print("Number of finished trials:", len(self.study.trials))
        print("Best trial parameters:", self.study.best_trial.params)
        print("Best score:", self.study.best_value)

        # Write config
        self.config["MODEL_PARAMS"][self.config["MODEL_NAME"]] = self.study.best_trial.params
        update_config(self.config)


if __name__ == "__main__":
    from finol.data_layer.DatasetLoader import DatasetLoader
    load_dataset_output = DatasetLoader().load_dataset()
    OptunaOptimizer(load_dataset_output=load_dataset_output).train_via_optuna()