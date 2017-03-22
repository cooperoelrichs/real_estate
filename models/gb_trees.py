import xgboost as xgb
from real_estate.models.price_model import PriceModel


class GBTrees(PriceModel):
    HAS_SIMPLE_COEFS = False
    HAS_FEATURE_IMPORTANCE = True

    PARAMS = {
        'max_depth': 12,  # 3
        'learning_rate': 0.08,  # 0.1
        'n_estimators': 1500,  # 100
        'silent': True,
        'objective': 'reg:linear',
        'nthread': -1,  # -1
        'gamma': 0,  # 0
        'min_child_weight': 1,  # 1
        'max_delta_step': 0,
        'subsample': 0.8,  # 1
        'colsample_bytree': 0.1,  # 1
        'colsample_bylevel': 1,
        'reg_alpha': 0,  # 0
        'reg_lambda': 1,  # 1
        'scale_pos_weight': 1,
        'base_score': 0.5,
        'seed': 0,
        'missing': None
    }

    def __init__(self, X, y, X_labels, df):
        self.model = GBTrees.make_model(self.PARAMS)
        self.setup_self(X, y, X_labels, df)

    def make_model(params):
        return xgb.XGBRegressor(**params)

    def feature_importance(self):
        return self.model.booster().get_fscore()
