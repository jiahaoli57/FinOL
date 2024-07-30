__version__ = "0.1.20"
__author__ = "FinOL Contributors"

# python setup.py sdist build
# twine upload dist/*

# from rich import print
from finol.utils import load_config, update_config, detect_device
config = load_config()
detect_device(config)
update_config(config)