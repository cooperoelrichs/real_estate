import xgboost as xgb
from real_estate.models.price_model import PriceModel


class GBTrees(PriceModel):
    MODEL_CLASS = xgb.XGBRegressor
    HAS_SIMPLE_COEFS = False
    HAS_FEATURE_IMPORTANCE = True

    PARAMS = {
        # max_depth 60, learning_rate 0.010, n_estimators 3000, reg_lambda 10: 0.743
        'booster': 'gbtree',
        'max_depth': 60,
        'learning_rate': 0.01,
        'n_estimators': 3000,
        'silent': True,
        'objective': 'reg:linear',
        'n_jobs': 6,
        'gamma': 0,
        'min_child_weight': 4,
        'max_delta_step': 0,
        'subsample': 0.5,
        'colsample_bytree': 1,
        'colsample_bylevel': 1,
        'reg_alpha': 1,
        'reg_lambda': 10,
        'scale_pos_weight': 1,
        'base_score': 0.5,
        'seed': 0,
        'missing': None
    }

    def __init__(self, X, y, X_labels, params=None):
        super().__init__(X, y, X_labels, params)

    def feature_importance(self):
        return self.model.feature_importances_
