import os
import subprocess

ROOT_PATH = "/root/miniconda3/envs/test/lib/python3.7/site-packages/finol"


# def download_data():
#     github_urls = [
#         "http://github.com/ai4finol/finol_data.git"
#         "git://github.com/ai4finol/finol_data.git",
#         "https://github.com/ai4finol/finol_data.git",
#     ]
#     for github_url in github_urls:
#         try:
#             # local_path = ROOT_PATH + r"\data"  # useless in Colab, so we use the following command
#             local_path = os.path.join(ROOT_PATH, "data")
#             subprocess.run(["git", "clone", github_url, local_path])
#             break
#         except subprocess.CalledProcessError as e:
#             print(f"Error downloading data from {github_url}: {e}")
#             continue

if __name__ == "__main__":
    # download_data()
    value = 3.14
    print(value.type())