import sys
import subprocess
import os
import time

import torch

import torch.nn.functional as F

from finol.update import *
from finol.config import *
from finol import __version__

ROOT_PATH = os.path.dirname(os.path.realpath(__file__))


def check_update():
    latest = get_latest_version()
    if __version__ == latest:
        print("The current FinOL is latest")
    else:
        print("The current FinOL is not latest, please update by `pip install --upgrade finol`")
        if GET_LATEST_FINOL:
            sys.exit()


def download_data():
    github_url = 'https://github.com/ai4finol/finol_data.git'
    local_path = ROOT_PATH + r'\data'
    subprocess.run(['git', 'clone', github_url, local_path])


def portfolio_selection(final_scores):
    portfolio = F.softmax(final_scores, dim=-1)
    return portfolio


def actual_portfolio_selection(final_scores):
    NUM_ASSETS = final_scores.shape[-1]
    if PROP_WINNERS == 1:
        portfolio = F.softmax(final_scores, dim=-1)
    else:
        NUM_SELECTED_ASSETS = int(NUM_ASSETS * PROP_WINNERS)
        assert NUM_SELECTED_ASSETS >= 0
        assert NUM_SELECTED_ASSETS <= NUM_ASSETS

        values, indices = torch.topk(final_scores, k=NUM_SELECTED_ASSETS)
        winners_mask = torch.ones_like(final_scores, device=DEVICE)
        winners_mask.scatter_(1, indices, 0).detach()
        portfolio = F.softmax(final_scores - 1e9 * winners_mask, dim=-1)
    return portfolio
