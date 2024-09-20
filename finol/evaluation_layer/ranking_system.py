import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from collections import defaultdict
from finol.utils import ROOT_PATH, load_config

config = load_config()

def compute_elo(battles, K=4, SCALE=400, BASE=10, INIT_RATING=1000):
    rating = defaultdict(lambda: INIT_RATING)

    for rd, model_a, model_b, winner in battles[['model_a', 'model_b', 'winner']].itertuples():
        ra = rating[model_a]
        rb = rating[model_b]
        ea = 1 / (1 + BASE ** ((rb - ra) / SCALE))
        eb = 1 / (1 + BASE ** ((ra - rb) / SCALE))
        if winner == "model_a":
            sa = 1
        elif winner == "model_b":
            sa = 0
        elif winner == "tie" or winner == "tie (bothbad)":
            sa = 0.5
        else:
            raise Exception(f"unexpected vote {winner}")
        rating[model_a] += K * (sa - ea)
        rating[model_b] += K * (1 - sa - eb)

    return rating


def preety_print_elo_ratings(ratings):
    df = pd.DataFrame([
        [n, ratings[n]] for n in ratings.keys()
    ], columns=["Model", "Elo rating"]).sort_values("Elo rating", ascending=False).reset_index(drop=True)
    df["Elo rating"] = (df["Elo rating"] + 0.5).astype(int)
    df.index = df.index + 1
    return df


def get_bootstrap_result(battles, func_compute_elo, num_round):
    rows = []
    for i in tqdm(range(num_round), desc="bootstrap"):
        rows.append(func_compute_elo(battles.sample(frac=1.0, replace=True)))
    df = pd.DataFrame(rows)
    return df[df.median().sort_values(ascending=False).index]


def visualize_bootstrap_scores(df, title):
    bars = pd.DataFrame(dict(
        lower=df.quantile(0.025),
        rating=df.quantile(0.5),
        upper=df.quantile(0.975)
    )).reset_index(names="model").sort_values("rating", ascending=False)

    bars['error_y'] = bars['upper'] - bars['rating']
    bars['error_y_minus'] = bars['rating'] - bars['lower']

    plt.figure(figsize=(12, 5))
    plt.errorbar(bars['model'], bars['rating'],
                 yerr=[bars['error_y_minus'], bars['error_y']],
                 fmt='o', markersize=3, capsize=5, color='gray', ecolor='gray')  # , label='Rating with CI'

    for i, txt in enumerate(np.round(bars['rating'], 2)):
        plt.text(bars['model'].iloc[i], bars['rating'].iloc[i], str(txt),
                 ha='center', va='bottom', fontsize=8)

    plt.title(title)
    plt.xlabel('Model')
    plt.ylabel('Rating')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    # plt.legend(loc="best")  # lower left upper left
    plt.savefig(ROOT_PATH + "/" + title + ".pdf", format="pdf", dpi=300, bbox_inches="tight")
    plt.show()
    plt.figure()


def compute_elo_mle(df, SCALE=400, BASE=10, INIT_RATING=1000):
    from sklearn.linear_model import LogisticRegression
    models = pd.concat([df["model_a"], df["model_b"]]).unique()
    models = pd.Series(np.arange(len(models)), index=models)
    p = len(models.index)
    n = df.shape[0]

    X = np.zeros([n, p])
    X[np.arange(n), models[df["model_a"]]] = +math.log(BASE)
    X[np.arange(n), models[df["model_b"]]] = -math.log(BASE)

    Y = np.zeros(n)
    Y[df["winner"] == "model_a"] = 1.0

    lr = LogisticRegression(fit_intercept=False)
    lr.fit(X,Y)

    elo_scores = SCALE * lr.coef_[0] + INIT_RATING

    return pd.Series(elo_scores, index=models.index).sort_values(ascending=False)



r = ROOT_PATH + r"\data\benchmark_results\profit_metrics"
NYSEO_df = pd.read_excel(r + r"\NYSE(O)\final_profit_result.xlsx", sheet_name="Sheet1")
NYSEN_df = pd.read_excel(r + r"\NYSE(N)\final_profit_result.xlsx", sheet_name="Sheet1")
DJIA_df = pd.read_excel(r + r"\DJIA\final_profit_result.xlsx", sheet_name="Sheet1")
SP500_df = pd.read_excel(r + r"\SP500\final_profit_result.xlsx", sheet_name="Sheet1")
TSE_df = pd.read_excel(r + r"\TSE\final_profit_result.xlsx", sheet_name="Sheet1")
SSE_df = pd.read_excel(r + r"\SSE\final_profit_result.xlsx", sheet_name="Sheet1")
HSI_df = pd.read_excel(r + r"\HSI\final_profit_result.xlsx", sheet_name="Sheet1")
CMEG_df = pd.read_excel(r + r"\CMEG\final_profit_result.xlsx", sheet_name="Sheet1")
CRYPTO_df = pd.read_excel(r + r"\CRYPTO\final_profit_result.xlsx", sheet_name="Sheet1")


data = {
    "Model": [
        "Market", "Best", "UCRP", "BCRP",
        "UP", "EG", "SCRP", "PPT", "SSPO",
        "ANTI1", "ANTI2", "PAMR", "CWMR-Var", "CWMR-Stdev", "OLMAR-S", "OLMAR-E", "RMR", "RPRT",
        "AICTR", "KTPT",
        "SP", "ONS", "GRW", "WAAS", "CW-OGD"
    ],
    "NYSEO_CW": list(NYSEO_df.loc[NYSEO_df["Metric"] == "CW"].iloc[0, 1:].values),
    "NYSEO_APY": list(NYSEO_df.loc[NYSEO_df["Metric"] == "APY"].iloc[0, 1:].values),
    "NYSEO_SR": list(NYSEO_df.loc[NYSEO_df["Metric"] == "SR"].iloc[0, 1:].values),

    "NYSEN_CW": list(NYSEN_df.loc[NYSEN_df["Metric"] == "CW"].iloc[0, 1:].values),
    "NYSEN_APY": list(NYSEN_df.loc[NYSEN_df["Metric"] == "APY"].iloc[0, 1:].values),
    "NYSEN_SR": list(NYSEN_df.loc[NYSEN_df["Metric"] == "SR"].iloc[0, 1:].values),

    "DJIA_CW": list(DJIA_df.loc[DJIA_df["Metric"] == "CW"].iloc[0, 1:].values),
    "DJIA_APY": list(DJIA_df.loc[DJIA_df["Metric"] == "APY"].iloc[0, 1:].values),
    "DJIA_SR": list(DJIA_df.loc[DJIA_df["Metric"] == "SR"].iloc[0, 1:].values),

    "SP500_CW": list(SP500_df.loc[SP500_df["Metric"] == "CW"].iloc[0, 1:].values),
    "SP500_APY": list(SP500_df.loc[SP500_df["Metric"] == "APY"].iloc[0, 1:].values),
    "SP500_SR": list(SP500_df.loc[SP500_df["Metric"] == "SR"].iloc[0, 1:].values),

    "TSE_CW": list(TSE_df.loc[TSE_df["Metric"] == "CW"].iloc[0, 1:].values),
    "TSE_APY": list(TSE_df.loc[TSE_df["Metric"] == "APY"].iloc[0, 1:].values),
    "TSE_SR": list(TSE_df.loc[TSE_df["Metric"] == "SR"].iloc[0, 1:].values),

    "SSE_CW": list(SSE_df.loc[SSE_df["Metric"] == "CW"].iloc[0, 1:].values),
    "SSE_APY": list(SSE_df.loc[SSE_df["Metric"] == "APY"].iloc[0, 1:].values),
    "SSE_SR": list(SSE_df.loc[SSE_df["Metric"] == "SR"].iloc[0, 1:].values),

    "HSI_CW": list(HSI_df.loc[HSI_df["Metric"] == "CW"].iloc[0, 1:].values),
    "HSI_APY": list(HSI_df.loc[HSI_df["Metric"] == "APY"].iloc[0, 1:].values),
    "HSI_SR": list(HSI_df.loc[HSI_df["Metric"] == "SR"].iloc[0, 1:].values),

    "CMEG_CW": list(CMEG_df.loc[CMEG_df["Metric"] == "CW"].iloc[0, 1:].values),
    "CMEG_APY": list(CMEG_df.loc[CMEG_df["Metric"] == "APY"].iloc[0, 1:].values),
    "CMEG_SR": list(CMEG_df.loc[CMEG_df["Metric"] == "SR"].iloc[0, 1:].values),

    "CRYPTO_CW": list(CRYPTO_df.loc[CRYPTO_df["Metric"] == "CW"].iloc[0, 1:].values),
    "CRYPTO_APY": list(CRYPTO_df.loc[CRYPTO_df["Metric"] == "APY"].iloc[0, 1:].values),
    "CRYPTO_SR": list(CRYPTO_df.loc[CRYPTO_df["Metric"] == "SR"].iloc[0, 1:].values),
}

models_df = pd.DataFrame(data)
models_df = models_df[~models_df['Model'].isin(['Best', 'BCRP'])]
print(models_df)

#
battles = []

for dataset in ["NYSEO", "NYSEN", "DJIA", "SP500", "TSE", "SSE", "HSI", "CMEG", "CRYPTO"]:
# for dataset in ['CRYPTO', 'CMEG', 'HSI', 'SSE', 'TSE', 'SP500', 'DJIA', 'NYSEN', 'NYSEO']:
    for index, row in models_df.iterrows():
        model_a = row["Model"]
        for inner_index, inner_row in models_df.iterrows():
            if index != inner_index:
                model_b = inner_row["Model"]
                for metric in ["CW", "APY", "SR"]:
                # for metric in ['SR', 'APY', 'CW']:
                    a_value = row[f"{dataset}_{metric}"]
                    b_value = inner_row[f"{dataset}_{metric}"]

                    if a_value > b_value:
                        winner = "model_a"
                    elif a_value < b_value:
                        winner = "model_b"
                    else:
                        winner = "tie"

                    battles.append({
                        "model_a": model_a,
                        "model_b": model_b,
                        "winner": winner,
                    })

battles = pd.DataFrame(battles)
print(battles)


elo_ratings = compute_elo(battles)
elo = preety_print_elo_ratings(elo_ratings)
print(
    "elo:\n",
    elo
)

### Compute Bootstrap Confidence Interavals for Elo Scores
# The previous linear update method may be sensitive to battle orders.
# Here, we use bootstrap to get a more stable versoion and estimate the confidence intervals as well.
BOOTSTRAP_ROUNDS = 1000
np.random.seed(config["MANUAL_SEED"])
bootstrap_elo_lu = get_bootstrap_result(battles, compute_elo, BOOTSTRAP_ROUNDS)
bootstrap_lu_median = bootstrap_elo_lu.median().reset_index().set_axis(["model", "Elo rating"], axis=1)
bootstrap_lu_median["Elo rating"] = (bootstrap_lu_median["Elo rating"] + 0.5).astype(int)
print(
    "elo bootstrap version:\n",
    bootstrap_lu_median
)
visualize_bootstrap_scores(bootstrap_elo_lu, "Bootstrap of Elo Estimates")


### Maximum Likelihood Estimation
# Another way to fit Elo ratings is using maximum likelihood estimation.
# Here, we provide an impelmentation with logistic regression.
# elo_mle_ratings = compute_elo_mle(battles)
# elo = preety_print_elo_ratings(elo_mle_ratings)
# print(
#     "elo bootstrap mle version:\n",
#     elo
# )
#
# elo_mle_bootstrap = get_bootstrap_result(battles, compute_elo_mle, BOOTSTRAP_ROUNDS)
# visualize_bootstrap_scores(elo_mle_bootstrap, "Bootstrap of MLE Elo Estimates")

####################################
# r = ROOT_PATH + r"\data\benchmark_results\risk_metrics"
# NYSEO_df = pd.read_excel(r + r"\NYSE(O)\final_risk_result.xlsx", sheet_name="Sheet1")
# NYSEN_df = pd.read_excel(r + r"\NYSE(N)\final_risk_result.xlsx", sheet_name="Sheet1")
# DJIA_df = pd.read_excel(r + r"\DJIA\final_risk_result.xlsx", sheet_name="Sheet1")
# SP500_df = pd.read_excel(r + r"\SP500\final_risk_result.xlsx", sheet_name="Sheet1")
# TSE_df = pd.read_excel(r + r"\TSE\final_risk_result.xlsx", sheet_name="Sheet1")
# SSE_df = pd.read_excel(r + r"\SSE\final_risk_result.xlsx", sheet_name="Sheet1")
# HSI_df = pd.read_excel(r + r"\HSI\final_risk_result.xlsx", sheet_name="Sheet1")
# CMEG_df = pd.read_excel(r + r"\CMEG\final_risk_result.xlsx", sheet_name="Sheet1")
# CRYPTO_df = pd.read_excel(r + r"\CRYPTO\final_risk_result.xlsx", sheet_name="Sheet1")
#
# data = {
#     "Model": [
#         "Market", "Best", "UCRP", "BCRP",
#         "UP", "EG", "SCRP", "PPT", "SSPO",
#         "ANTI1", "ANTI2", "PAMR", "CWMR-Var", "CWMR-Stdev", "OLMAR-S", "OLMAR-E", "RMR", "RPRT",
#         "AICTR", "KTPT",
#         "SP", "ONS", "GRW", "WAAS", "CW-OGD"
#     ],
#     "NYSEO_VR": list(NYSEO_df.loc[NYSEO_df["Metric"] == "VR"].iloc[0, 1:].values),
#     "NYSEO_MDD": list(NYSEO_df.loc[NYSEO_df["Metric"] == "MDD"].iloc[0, 1:].values),
#
#     "NYSEN_VR": list(NYSEN_df.loc[NYSEN_df["Metric"] == "VR"].iloc[0, 1:].values),
#     "NYSEN_MDD": list(NYSEN_df.loc[NYSEN_df["Metric"] == "MDD"].iloc[0, 1:].values),
#
#     "DJIA_VR": list(DJIA_df.loc[DJIA_df["Metric"] == "VR"].iloc[0, 1:].values),
#     "DJIA_MDD": list(DJIA_df.loc[DJIA_df["Metric"] == "MDD"].iloc[0, 1:].values),
#
#     "SP500_VR": list(SP500_df.loc[SP500_df["Metric"] == "VR"].iloc[0, 1:].values),
#     "SP500_MDD": list(SP500_df.loc[SP500_df["Metric"] == "MDD"].iloc[0, 1:].values),
#
#     "TSE_VR": list(TSE_df.loc[TSE_df["Metric"] == "VR"].iloc[0, 1:].values),
#     "TSE_MDD": list(TSE_df.loc[TSE_df["Metric"] == "MDD"].iloc[0, 1:].values),
#
#     "SSE_VR": list(SSE_df.loc[SSE_df["Metric"] == "VR"].iloc[0, 1:].values),
#     "SSE_MDD": list(SSE_df.loc[SSE_df["Metric"] == "MDD"].iloc[0, 1:].values),
#
#     "HSI_VR": list(HSI_df.loc[HSI_df["Metric"] == "VR"].iloc[0, 1:].values),
#     "HSI_MDD": list(HSI_df.loc[HSI_df["Metric"] == "MDD"].iloc[0, 1:].values),
#
#     "CMEG_VR": list(CMEG_df.loc[CMEG_df["Metric"] == "VR"].iloc[0, 1:].values),
#     "CMEG_MDD": list(CMEG_df.loc[CMEG_df["Metric"] == "MDD"].iloc[0, 1:].values),
#
#     "CRYPTO_VR": list(CRYPTO_df.loc[CRYPTO_df["Metric"] == "VR"].iloc[0, 1:].values),
#     "CRYPTO_MDD": list(CRYPTO_df.loc[CRYPTO_df["Metric"] == "MDD"].iloc[0, 1:].values),
# }
#
# models_df = pd.DataFrame(data)
# models_df = models_df[~models_df['Model'].isin(['Best', 'BCRP'])]
# print(models_df)
#
# battles = []
#
# for dataset in ["NYSEO", "NYSEN", "DJIA", "SP500", "TSE", "SSE", "HSI", "CMEG", "CRYPTO"]:
# # for dataset in ['CRYPTO', 'CMEG', 'HSI', 'SSE', 'TSE', 'SP500', 'DJIA', 'NYSEN', 'NYSEO']:
#     for index, row in models_df.iterrows():
#         model_a = row["Model"]
#         for inner_index, inner_row in models_df.iterrows():
#             if index != inner_index:
#                 model_b = inner_row["Model"]
#                 for metric in ["VR", "MDD"]:
#                 # for metric in ["MDD", "VR"]:
#                     a_value = row[f"{dataset}_{metric}"]
#                     b_value = inner_row[f"{dataset}_{metric}"]
#
#                     if a_value < b_value:
#                         winner = "model_a"
#                     elif a_value > b_value:
#                         winner = "model_b"
#                     else:
#                         winner = "tie"
#
#                     battles.append({
#                         "model_a": model_a,
#                         "model_b": model_b,
#                         "winner": winner,
#                     })
#
#
# battles = pd.DataFrame(battles)
# print(battles)
#
# elo_ratings = compute_elo(battles)
# elo = preety_print_elo_ratings(elo_ratings)
# print(
#     "elo:\n",
#     elo
# )
#
# ### Compute Bootstrap Confidence Interavals for Elo Scores
# # The previous linear update method may be sensitive to battle orders.
# # Here, we use bootstrap to get a more stable versoion and estimate the confidence intervals as well.
# BOOTSTRAP_ROUNDS = 1000
# np.random.seed(config["MANUAL_SEED"])
# bootstrap_elo_lu = get_bootstrap_result(battles, compute_elo, BOOTSTRAP_ROUNDS)
# bootstrap_lu_median = bootstrap_elo_lu.median().reset_index().set_axis(["model", "Elo rating"], axis=1)
# bootstrap_lu_median["Elo rating"] = (bootstrap_lu_median["Elo rating"] + 0.5).astype(int)
# print(
#     "elo bootstrap version:\n",
#     bootstrap_lu_median
# )
# visualize_bootstrap_scores(bootstrap_elo_lu, "Bootstrap of Elo Estimates")
#
#
# ### Maximum Likelihood Estimation
# # Another way to fit Elo ratings is using maximum likelihood estimation.
# # Here, we provide an impelmentation with logistic regression.
# elo_mle_ratings = compute_elo_mle(battles)
# elo = preety_print_elo_ratings(elo_mle_ratings)
# print(
#     "elo bootstrap mle version:\n",
#     elo
# )
#
# elo_mle_bootstrap = get_bootstrap_result(battles, compute_elo_mle, BOOTSTRAP_ROUNDS)
# visualize_bootstrap_scores(elo_mle_bootstrap, "Bootstrap of MLE Elo Estimates")
