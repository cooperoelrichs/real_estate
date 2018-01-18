import xgboost as xgb
from real_estate.models.price_model import PriceModel


class GBTrees(PriceModel):
    HAS_SIMPLE_COEFS = False
    HAS_FEATURE_IMPORTANCE = True

    PARAMS = {
        'booster': 'gbtree',
        'max_depth': 20,  # 3
        'learning_rate': 0.1,  # 0.1
        'n_estimators': 1000,  # 100
        'silent': True,
        'objective': 'reg:linear',
        'nthread': -1,  # -1
        'gamma': 0,  # 0
        'min_child_weight': 4,  # 1
        'max_delta_step': 0,
        'subsample': 0.5,  # 1
        'colsample_bytree': 1,  # 1
        'colsample_bylevel': 1,
        'reg_alpha': 0,  # L1 regularization parameter
        'reg_lambda': 1,  # L2 regularization parameter
        'scale_pos_weight': 1,
        'base_score': 0.5,
        'seed': 0,
        'missing': None
    }

    def __init__(self, X, y, X_labels, params=None):
        if params == None:
            self.model = GBTrees.make_model(self.PARAMS)
        else:
            self.model = GBTrees.make_model(params)
        self.setup_self(X, y, X_labels)

    def make_model(params):
        return xgb.XGBRegressor(**params)

    def feature_importance(self):
        return self.model.feature_importances_
