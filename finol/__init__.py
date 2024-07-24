__version__ = "0.1.12"
__author__ = "FinOL contributors"

# python setup.py sdist build
# twine upload dist/*

from finol.utils import check_update, download_data
check_update()
download_data()