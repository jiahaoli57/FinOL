import sys
import subprocess
import os

from finol.update import get_latest_version
from finol import __version__
# from finol.config import *


ROOT_PATH = os.path.dirname(os.path.realpath(__file__))

def check_update(GET_LATEST_FINOL):
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

