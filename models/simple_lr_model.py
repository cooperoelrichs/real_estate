from sklearn.linear_model import LinearRegression, Ridge
from real_estate.models.price_model import PriceModel


class LinearModel(PriceModel):
    MODEL_CLASS = LinearRegression
    HAS_SIMPLE_COEFS = True
    HAS_FEATURE_IMPORTANCE = False

    PARAMS = {
        'fit_intercept': True,
        'normalize': False,
        'copy_X': True,
        'n_jobs': -1
    }

    def __init__(self, X, y, X_labels, params=None):
        super().__init__(X, y, X_labels, params)


class RidgeModel(PriceModel):
    MODEL_CLASS = Ridge
    HAS_SIMPLE_COEFS = True
    HAS_FEATURE_IMPORTANCE = False
    # MODEL_APLHA = 0.1

    PARAMS = {
        'alpha': 0.1,
        'fit_intercept': True,
        'normalize': False,
        'copy_X': True,
    }

    def __init__(self, X, y, X_labels, params=None):
        super().__init__(X, y, X_labels, params)
