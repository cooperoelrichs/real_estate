import xgboost as xgb
from real_estate.models.price_model import PriceModel


class GBTrees(PriceModel):
    HAS_SIMPLE_COEFS = False
    HAS_FEATURE_IMPORTANCE = True

    def __init__(self, X, y, X_labels, df):
        self.model = xgb.XGBRegressor(
            max_depth=3,  # 3
            learning_rate=0.1,  # 0.1
            n_estimators=500,  # 100
            silent=True,
            objective='reg:linear',
            nthread=-1,
            gamma=0,  # 0
            min_child_weight=1,  # 1
            max_delta_step=0,
            subsample=1,  # 1
            colsample_bytree=1,  # 1
            colsample_bylevel=1,
            reg_alpha=0,  # 0
            reg_lambda=1,  # 1
            scale_pos_weight=1,
            base_score=0.5,
            seed=0,
            missing=None
        )
        self.setup_self(X, y, X_labels, df)

    def feature_importance(self):
        return self.model.booster().get_fscore()
