import time
import torch
import os
import matplotlib

import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import pandas as pd
from shutil import copy2
from IPython import display
from tqdm import tqdm

from finol.data_layer.data_loader import *
from finol.model_layer.model_selector import *
from finol.optimization_layer.criterion_selector import *
from finol.optimization_layer.optimizer_selector import *
from finol.evaluation_layer.benchmark_loader import *
from finol.config import *

logdir = PARENT_PATH + '/logdir/' + str(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))

is_ipython = 'inline' in matplotlib.get_backend()
metadata = [
    ('TUTORIAL_MODE', TUTORIAL_MODE),
    ('TUTORIAL_NAME', TUTORIAL_NAME),
    ('Dataset', DATASET_NAME),
    ('TRAIN_END_TIMESTAMP', DATASET_SPLIT_CONFIG.get(DATASET_NAME)["TRAIN_END_TIMESTAMP"]),
    ('SCALER', SCALER),
    ('BATCH_SIZE', BATCH_SIZE[DATASET_NAME]),
    ('MODEL_NAME', MODEL_NAME),
    ('MODEL_CONFIG', MODEL_CONFIG[MODEL_NAME]),
    ('DROPOUT', DROPOUT),
    ('OPTIMIZER_NAME', OPTIMIZER_NAME),
    ('LEARNING_RATE', LEARNING_RATE),
    ('CRITERION_NAME', CRITERION_NAME),
    ('LAMBDA_L2', LAMBDA_L2),
    ('DEVICE', DEVICE),
    ('NUM_EPOCHES', NUM_EPOCHES)
]


def plot_loss_notebook(train_loss_list, val_loss_list):
    if is_ipython:
        plt.ion()
        plt.figure(figsize=(12, 5))
        plt.clf()
        plt.plot(train_loss_list, linestyle='-', marker=MARKERS[0], markevery=MARKEVERY, color='black', alpha=ALPHA, label='train loss')
        plt.plot(val_loss_list, linestyle=':', marker=MARKERS[1], markevery=MARKEVERY, color='black', alpha=ALPHA, label='val loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        y_start = 1.0
        y_step = -0.05

        for i, (label, value) in enumerate(metadata):
            y_coord = y_start + i * y_step
            plt.text(1.05, y_coord, f'{label}: {value}', transform=plt.gca().transAxes, va='center', ha='left')

        plt.legend()
        plt.grid(True)
        display.clear_output(wait=True)
        display.display(plt.gcf())
        plt.tight_layout()
        plt.savefig(logdir + '/loss_' + DATASET_NAME + '.png')
        plt.clf()
        plt.close()


def plot_loss(train_loss_list, val_loss_list):
    # not_ipython = 'inline' not in matplotlib.get_backend()
    if not is_ipython:
        plt.figure(figsize=(12, 5))
        plt.plot(np.array(train_loss_list), linestyle='-', marker=MARKERS[0], markevery=MARKEVERY, color='black', alpha=ALPHA, label='train loss')
        plt.plot(np.array(val_loss_list), linestyle=':', marker=MARKERS[1], markevery=MARKEVERY, color='black', alpha=ALPHA, label='val loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        y_start = 1.0
        y_step = -0.05

        for i, (label, value) in enumerate(metadata):
            y_coord = y_start + i * y_step
            plt.text(1.05, y_coord, f'{label}: {value}', transform=plt.gca().transAxes, va='center', ha='left')

        plt.tight_layout()
        plt.savefig(logdir + '/loss_' + DATASET_NAME + '.png')
        plt.show()


def train_model(load_dataset_output):
    torch.manual_seed(MANUAL_SEED)
    os.makedirs(logdir)
    copy2(ROOT_PATH + '/config.py', logdir)

    print(
        load_dataset_output
    )
    train_loader = load_dataset_output['train_loader']
    val_loader = load_dataset_output['val_loader']
    test_loader = load_dataset_output['test_loader']
    NUM_TRAIN_PERIODS = load_dataset_output['NUM_TRAIN_PERIODS']
    NUM_VAL_PERIODS = load_dataset_output['NUM_VAL_PERIODS']
    NUM_TEST_PERIODS = load_dataset_output['NUM_TEST_PERIODS']

    model = select_model(load_dataset_output)
    optimizer = select_optimizer(model)
    criterion = select_criterion()

    train_loss_list = []
    val_loss_list = []
    best_val_loss = float('inf')

    for e in tqdm(range(NUM_EPOCHES), desc="Training"):
        model.train()
        train_loss = 0
        for i, data in enumerate(train_loader, 1):
            x_data, label = data
            portfolio = model(x_data.float())
            loss = criterion(portfolio, label.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_loss_list.append(train_loss)

        with torch.no_grad():
            model.eval()
            val_loss = 0
            for i, data in enumerate(val_loader, 1):
                val_data, label = data
                portfolio = model(val_data.float())
                loss = criterion(portfolio, label.float())
                val_loss += loss.item()

            val_loss /= len(val_loader)
            val_loss_list.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model, logdir + '/best_model_'+DATASET_NAME+'.pt')
            torch.save(model, logdir + '/last_model_' + DATASET_NAME + '.pt')

        if (e + 1) % 10 == 0:
            print('Epoch: {}, Train Loss: {}, Val Loss: {}'.format(e + 1, train_loss, val_loss))
            plot_loss_notebook(train_loss_list, val_loss_list)

    plot_loss(train_loss_list, val_loss_list)

    train_model_output = {
        "last_model": torch.load(logdir + '/last_model_'+DATASET_NAME+'.pt'),
        "best_model": torch.load(logdir + '/best_model_'+DATASET_NAME+'.pt'),
        "logdir": logdir,
    }
    return train_model_output


if __name__ == '__main__':
    load_dataset_output = load_dataset()
    train_model_output = train_model(load_dataset_output)