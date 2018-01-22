from sklearn.linear_model import LinearRegression, Ridge
from real_estate.models.price_model import PriceModel


class LinearModel(PriceModel):
    HAS_SIMPLE_COEFS = True
    HAS_FEATURE_IMPORTANCE = False

    PARAMS = {
        'fit_intercept': True,
        'normalize': False,
        'copy_X': True,
        'n_jobs': -1
    }

    def __init__(self, X, y, X_labels, params=None):
        if params == None:
            self.model = LinearModel.make_model(self.PARAMS)
        else:
            self.model = LinearModel.make_model(params)

        self.setup_self(X, y, X_labels)

    def make_model(params):
        return LinearRegression(**params)


class RidgeModel(PriceModel):
    HAS_SIMPLE_COEFS = True
    HAS_FEATURE_IMPORTANCE = False
    MODEL_APLHA = 0.1

    def __init__(self, X, y, X_labels):
        self.model = Ridge(
            alpha=self.MODEL_APLHA,
            fit_intercept=True,
            normalize=False,
            copy_X=True,
        )

        self.setup_self(X, y, X_labels)
