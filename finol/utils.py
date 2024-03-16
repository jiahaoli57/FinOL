from finol.update import get_latest_version
from finol import __version__

import sys
from finol.config import *


def check_update():
    latest = get_latest_version()
    if __version__ == latest:
        print("The current FinOL is latest")
    else:
        print("The current FinOL is not latest, please update by `pip install --upgrade finol`")
        if GET_LATEST_FINOL:
            sys.exit()