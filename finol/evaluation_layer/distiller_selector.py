from sklearn.linear_model import *
from finol.config import *

distiller_dict = {
    ###############################
    # Classical linear regressors #
    ###############################
    'LinearRegression': LinearRegression,  # Ordinary least squares Linear Regression
    'Ridge': Ridge,  # Linear least squares with l2 regularization
    'RidgeCV': RidgeCV,  # Ridge regression with built-in cross-validation
    'SGDRegressor': SGDRegressor,  # Linear model fitted by minimizing a regularized empirical loss with SGD

    ######################################
    # Regressors with variable selection #
    ######################################
    'ElasticNet': ElasticNet,  # Linear regression with combined L1 and L2 priors as regularizer
    'ElasticNetCV': ElasticNetCV,  # Elastic Net model with iterative fitting along a regularization path
    'Lars': Lars,  # Least Angle Regression model a.k.a
    'LarsCV': LarsCV,  # Cross-validated Least Angle Regression model
    'Lasso': Lasso,  # Linear Model trained with L1 prior as regularizer (aka the Lasso)
    'LassoCV': LassoCV,  # Lasso linear model with iterative fitting along a regularization path
    'LassoLars': LassoLars,  # Lasso model fit with Least Angle Regression a.k.a
    'LassoLarsCV': LassoLarsCV,  # Cross-validated Lasso, using the LARS algorithm
    'LassoLarsIC': LassoLarsIC,  # Lasso model fit with Lars using BIC or AIC for model selection
    'OrthogonalMatchingPursuit': OrthogonalMatchingPursuit,  # Orthogonal Matching Pursuit model (OMP)
    'OrthogonalMatchingPursuitCV': OrthogonalMatchingPursuitCV,  # Cross-validated Orthogonal Matching Pursuit model (OMP)

    #######################
    # Bayesian regressors #
    #######################
    'ARDRegression': ARDRegression,  # Bayesian ARD regression
    'BayesianRidge': BayesianRidge,  # Bayesian ridge regression

    #############################
    # Outlier-robust regressors #
    #############################
    'HuberRegressor': HuberRegressor,  # L2-regularized linear regression model that is robust to outliers
    'QuantileRegressor': QuantileRegressor,  # Linear regression model that predicts conditional quantiles
    'RANSACRegressor': RANSACRegressor,  # RANSAC (RANdom SAmple Consensus) algorithm
    'TheilSenRegressor': TheilSenRegressor,  # Theil-Sen Estimator: robust multivariate regression model

    ##################################################
    # Generalized linear models (GLM) for regression #
    ##################################################
    'PoissonRegressor': PoissonRegressor,  # Generalized Linear Model with a Poisson distribution
    'TweedieRegressor': TweedieRegressor,  # Generalized Linear Model with a Tweedie distribution
    'GammaRegressor': GammaRegressor,  # Generalized Linear Model with a Gamma distribution

    #################
    # Miscellaneous #
    #################
    'PassiveAggressiveRegressor': PassiveAggressiveRegressor,  # Passive Aggressive Regressor
}


def select_distiller():
    distiller_cls = distiller_dict.get(INTERPRETABLE_ANALYSIS_CONFIG['DISTILLER_NAME'], None)
    if distiller_cls is None:
        raise ValueError(f"Invalid distiller: {INTERPRETABLE_ANALYSIS_CONFIG['DISTILLER_NAME']}. Supported distillers are: {distiller_dict}")
    return distiller_cls()

# fit_intercept=True, alpha=0.0001
