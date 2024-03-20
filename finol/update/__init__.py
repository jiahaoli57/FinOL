import requests


def get_latest_version():
    response = requests.get('https://pypi.org/pypi/finol/json')
    latest_version = response.json()['info']['version']
    return latest_version