import sys
import subprocess
import os
import time
import json
import requests
import torch
import random
import numpy as np
import torch.nn.functional as F

from shutil import copy2
from finol.update import get_latest_version
from finol import __version__


ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
PARENT_PATH = os.path.dirname(ROOT_PATH)

print("ROOT_PATH", ROOT_PATH)
print("PARENT_PATH", PARENT_PATH)
print()


def check_update():
    latest = get_latest_version()
    if __version__ == latest:
        print(f"The current FinOL (version: {__version__}) is latest")
    else:
        print(f"The current FinOL (version: {__version__}) is not latest, "
              f"The latest version on https://pypi.org/project/finol is {latest}, "
              f"please consider updating by `pip install --upgrade finol`")
        # print("Before updating, remember to back up any modifications you made to the FinOL project, such as added model code.")
        # print("Note that `pip install --upgrade finol` will overwrite all files except the `logdir` folder, so you don't need to back up the `logdir`.")
        # sys.exit()


def make_logdir():
    # logdir = PARENT_PATH + "/logdir/" + str(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))
    logdir_path = os.path.join(ROOT_PATH, "logdir/")
    logdir = logdir_path + str(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))
    os.makedirs(logdir)
    copy2(ROOT_PATH + "/config.json", logdir)
    return logdir


def load_config():
    with open(ROOT_PATH + "/config.json", "r") as f:
        config = json.load(f)
    return config


def update_config(config):
    with open(ROOT_PATH + "/config.json", "w") as f:
        json.dump(config, f, indent=4)


def detect_device(config):
    if config["DEVICE"] == "auto":
        # Detect available device
        if torch.cuda.is_available():
            config["DEVICE"] = "cuda"
        else:
            config["DEVICE"] = "cpu"
    return config


def download_data():
    # github_url = "https://github.com/ai4finol/finol_data.git"
    # github_url = "git://github.com/ai4finol/finol_data.git"
    # github_url = "http://github.com/ai4finol/finol_data.git"
    # local_path = ROOT_PATH + r"\data"  # useless in Colab, so we use the following command
    # local_path = os.path.join(ROOT_PATH, "data")
    # subprocess.run(["git", "clone", github_url, local_path])

    github_urls = [
        "http://github.com/ai4finol/finol_data.git"
        "git://github.com/ai4finol/finol_data.git",
        "https://github.com/ai4finol/finol_data.git",
    ]
    for github_url in github_urls:
        try:
            # local_path = ROOT_PATH + r"\data"  # useless in Colab, so we use the following command
            local_path = os.path.join(ROOT_PATH, "data")
            subprocess.run(["git", "clone", github_url, local_path])
            break
        except subprocess.CalledProcessError as e:
            print(f"Error downloading data from {github_url}: {e}")
            continue


def portfolio_selection(final_scores):
    portfolio = F.softmax(final_scores, dim=-1)
    # print(final_scores)
    # print(portfolio)
    assert torch.all(portfolio >= 0), "Portfolio contains non-negative values."
    return portfolio


def actual_portfolio_selection(final_scores):
    config = load_config()
    PROP_WINNERS = config["PROP_WINNERS"]
    DEVICE = config["DEVICE"]

    NUM_ASSETS = final_scores.shape[-1]
    if PROP_WINNERS == 1:
        portfolio = F.softmax(final_scores, dim=-1)
    else:
        NUM_SELECTED_ASSETS = int(NUM_ASSETS * PROP_WINNERS)
        assert NUM_SELECTED_ASSETS > 0
        assert NUM_SELECTED_ASSETS <= NUM_ASSETS

        values, indices = torch.topk(final_scores, k=NUM_SELECTED_ASSETS)
        winners_mask = torch.ones_like(final_scores, device=DEVICE)
        winners_mask.scatter_(1, indices, 0).detach()
        portfolio = F.softmax(final_scores - 1e9 * winners_mask, dim=-1)
    assert torch.all(portfolio >= 0), "Portfolio contains non-negative values."
    return portfolio


def send_message_dingding(dingding_message):
    headers_dingding = {
        "Content-Type": "application/json",
        "Charset": "UTF-8"
    }
    content = f"\n" \
              f"message：\tmodel finish training & testing\n" \
              f"logdir：\t{dingding_message['logdir']}\n" \
              f"CW：\t{dingding_message['CW']}\n" \

    webhook = "https://oapi.dingtalk.com/robot/send?access_token=xxxxxxxxxxxxxxxxxxxxxxxx"  # your own dingding api
    message = {
        "msgtype": "text",
        "text": {
            "content": content
        },
        "at": {
            "isAtAll": True
        }
    }
    message_json = json.dumps(message)
    try:
        info = requests.post(url=webhook, data=message_json, headers=headers_dingding, verify=False).json()
    except Exception as e:
        print(f"{e}")
        return
    if info.get("errcode") != 0:
        print(f"{info}")


def get_variable_name(var):
    return next(key for key, value in globals().items() if value is var)


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True