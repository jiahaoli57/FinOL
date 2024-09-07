import time
import torch

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# from IPython import display
from tqdm import tqdm
from typing import List, Tuple, Dict, Union
from finol.model_layer.model_instantiator import ModelInstantiator
from finol.optimization_layer.criterion_selector import CriterionSelector
from finol.optimization_layer.optimizer_selector import OptimizerSelector
from finol.optimization_layer.parameters_tuner import ParametersTuner
from finol.utils import load_config, portfolio_selection, set_seed, add_prefix


class ModelTrainer:
    """
    Class to train a machine learning model for portfolio selection.

    :param load_dataset_output: Dictionary containing output from function :func:`~finol.data_layer.DatasetLoader.load_dataset`.
    """
    def __init__(self, load_dataset_output: Dict) -> None:
        self.config = load_config()
        self.load_dataset_output = load_dataset_output
        self.logdir = load_dataset_output["logdir"]
        self.train_loader = load_dataset_output["train_loader"]
        self.val_loader = load_dataset_output["val_loader"]
        self.test_loader = load_dataset_output["test_loader_"]

        # self.is_ipython = "inline" in matplotlib.get_backend()
        # print(self.is_ipython)

    # def plot_loss_notebook(self) -> None:
    #     """
    #     Plot the training and validation losses in a notebook environment.
    #
    #     This method plots the average training and validation losses over epochs using matplotlib in an interactive notebook environment.
    #
    #     :return: None
    #     """
    #     if self.is_ipython:
    #         plt.ion()
    #         plt.figure()  # figsize=(12, 5)
    #         plt.clf()
    #         plt.plot(self.avg_train_loss_list, linestyle="-", marker=self.config["MARKERS"][0], markevery=int(self.config["NUM_EPOCHES"]/20), color="black", alpha=0.5, label="train loss")
    #         plt.plot(self.avg_val_loss_list, linestyle=":", marker=self.config["MARKERS"][1], markevery=int(self.config["NUM_EPOCHES"]/20), color="black", alpha=0.5, label="val loss")
    #         plt.xlabel("Epoch")
    #         plt.ylabel("Loss")
    #         plt.legend()
    #         plt.grid(True)
    #         display.clear_output(wait=True)
    #         display.display(plt.gcf())
    #         plt.tight_layout()
    #         # plt.yscale("log")
    #         plt.savefig(self.logdir + "/" + add_prefix("loss.pdf"), format="pdf", dpi=300, bbox_inches="tight")
    #         plt.clf()
    #         plt.close()

    def plot_loss(self) -> None:
        """
        Plot the training and validation losses.

        This method plots the average training and validation losses over epochs using matplotlib.

        :return: None
        """
        # not_ipython = "inline" not in matplotlib.get_backend()
        # if not self.is_ipython:
        plt.figure()  # figsize=(12, 5)
        plt.plot(np.array(self.avg_train_loss_list), linestyle="-", marker=self.config["MARKERS"][0], color="black", alpha=0.5, label="train loss")
        plt.plot(np.array(self.avg_val_loss_list), linestyle=":", marker=self.config["MARKERS"][1], color="black", alpha=0.5, label="val loss")
        # markevery = int(self.config["NUM_EPOCHES"] / 20)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        # plt.yscale("log")
        plt.savefig(self.logdir + "/" + add_prefix("loss.pdf"), format="pdf", dpi=300, bbox_inches="tight")
        plt.show()

        # from io import BytesIO
        # buffer = BytesIO()
        # plt.savefig(buffer, format='png')
        # buffer.seek(0)

    def train_model(self) -> Dict:
        """
        Train the machine learning model using the specified dataset.

        This method iterates over the specified number of epochs and trains the model using the training data.
        It calculates and stores the training and validation losses, saves the best model based on validation loss,
        and optionally plots the dynamic loss during training.

        :return: Dictionary containing the log directory for the trained model.
        """
        if self.config["TUNE_PARAMETERS"]:
            ParametersTuner(load_dataset_output=self.load_dataset_output).tune_parameters()
            pass

        set_seed(seed=self.config["MANUAL_SEED"])
        model = ModelInstantiator(self.load_dataset_output).instantiate_model()
        optimizer = OptimizerSelector(model).select_optimizer()
        criterion = CriterionSelector()

        self.train_loss_list = []
        self.val_loss_list = []
        self.avg_train_loss_list = []
        self.avg_val_loss_list = []
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
            self.train_loss_list.append(train_loss)
            self.avg_train_loss_list.append(sum(self.train_loss_list) / len(self.train_loss_list))

            if (e + 1) % self.config["SAVE_EVERY"] == 0:
            # if True:
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
                    self.val_loss_list.append(val_loss)
                    self.avg_val_loss_list.append(sum(self.val_loss_list) / len(self.val_loss_list))

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        # best_model = model
                        # print("best_model", "e:", e)
                        torch.save(model, self.logdir + "/" + add_prefix("best_model.pt"))

            # if self.config["PLOT_DYNAMIC_LOSS"]:
            #     if (e + 1) % 10 == 0:
            #         # print("Epoch: {}, Train Loss: {}, Val Loss: {}".format(e + 1, train_loss, val_loss))
            #         self.plot_loss_notebook()

        # self.plot_loss_notebook()
        self.plot_loss()

        train_model_output = {
            "logdir": self.logdir,
            # "last_model": model,
            # "best_model": best_model
        }
        # print(
        #     best_model
        # )
        torch.save(model, self.logdir + "/" + add_prefix("last_model.pt"))
        return train_model_output


if __name__ == "__main__":
    from finol.data_layer.dataset_loader import DatasetLoader
    from finol.evaluation_layer.model_evaluator import ModelEvaluator
    load_dataset_output = DatasetLoader().load_dataset()
    train_model_output = ModelTrainer(load_dataset_output).train_model()
    evaluate_model_output = ModelEvaluator(load_dataset_output, train_model_output).evaluate_model()