from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor)
from real_estate.models.price_model import PriceModel


N_ESTIMATORS = 3000
LEARNING_RATE = 0.01
MAX_DEPTH = 60
SUBSAMPLE = 0.5
BOOTSTRAP = True
OOB_SCORE = True


class GBR(PriceModel):
    MODEL_CLASS = GradientBoostingRegressor
    HAS_SIMPLE_COEFS = False
    HAS_FEATURE_IMPORTANCE = True

    PARAMS = {
        'loss': 'lad',  # ls
        'learning_rate': LEARNING_RATE,
        'n_estimators': N_ESTIMATORS,
        'max_depth': MAX_DEPTH,
        'subsample': SUBSAMPLE,
        'min_samples_split': 2,
        'min_samples_leaf': 1,

        # 'criterion': 'friedman_mse',
        'min_weight_fraction_leaf': 0.0,
        # 'min_impurity_decrease': 0.0,
        # 'min_impurity_split': None,
        'init': None,
        'random_state': None,
        'max_features': None,
        'alpha': 0.9,
        'verbose': 0,
        'max_leaf_nodes': None,
        'warm_start': False,
        'presort': 'auto'
    }

    def __init__(self, X, y, X_labels, params=None):
        super().__init__(X, y, X_labels, params)


class ETR(PriceModel):
    MODEL_CLASS = ExtraTreesRegressor
    HAS_SIMPLE_COEFS = False
    HAS_FEATURE_IMPORTANCE = True

    PARAMS = {
        'n_estimators': N_ESTIMATORS,
        # 'criterion': 'mse',
        'max_depth': MAX_DEPTH,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'min_weight_fraction_leaf': 0.0,
        'max_features': 'auto',
        'max_leaf_nodes': None,
        # 'min_impurity_decrease': 0.0,
        # 'min_impurity_split': None,
        'bootstrap': BOOTSTRAP,  # True
        'oob_score': OOB_SCORE,
        'n_jobs': 4,
        'random_state': None,
        'verbose': 0,
        'warm_start': False
    }

    def __init__(self, X, y, X_labels, params=None):
        super().__init__(X, y, X_labels, params)

    def feature_importance(self):
        raise NotImplementedError()


class RFR(PriceModel):
    MODEL_CLASS = RandomForestRegressor
    HAS_SIMPLE_COEFS = False
    HAS_FEATURE_IMPORTANCE = True

    PARAMS = {
        'n_estimators': N_ESTIMATORS,
        # 'criterion': 'mse',
        'max_depth': MAX_DEPTH,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'min_weight_fraction_leaf': 0.0,
        'max_features': 'auto',
        'max_leaf_nodes': None,
        # 'min_impurity_decrease': 0.0,
        # 'min_impurity_split': None,
        'bootstrap': BOOTSTRAP,
        'oob_score': OOB_SCORE,
        'n_jobs': 4,
        'random_state': None,
        'verbose': 0,
        'warm_start': False
    }

    def __init__(self, X, y, X_labels, params=None):
        super().__init__(X, y, X_labels, params)

    def feature_importance(self):
        raise NotImplementedError()
