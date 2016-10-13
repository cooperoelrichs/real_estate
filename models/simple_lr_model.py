import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation


class XY(object):
    """Generate X and y tensors from the properties DataFrame."""

    X_DATA = [
        ('bedrooms', 'continuous'),
        ('bathrooms', 'continuous'),
        ('garage_spaces', 'continuous'),
        ('property_type', 'categorical'),
        ('suburb', 'categorical')
    ]

    def __init__(self, df):
        df = self.filter_data(df)
        self.y = self.make_y(df)
        X = self.make_x(df)
        self.X = X.values
        self.X_columns = X.columns

    def filter_data(self, df):
        # Required data: sale_type, price, and suburb
        df = df[
            (
                (df['sale_type'] == 'Private Treaty') |
                (df['sale_type'] == 'Maybe Private Treaty')
            ) &
            (np.isfinite(df['price_min']) | np.isfinite(df['price_max'])) &
            pd.notnull(df['suburb']) &
            np.isfinite(df['bedrooms']) &
            np.isfinite(df['bathrooms']) &
            np.isfinite(df['garage_spaces'])
        ]
        return df

    def make_y(self, df):
        return df[['price_min', 'price_max']].values.mean(axis=1)

    def make_x(self, df):
        X = df[[a for a, _ in self.X_DATA]]
        cats = [a for a, b in self.X_DATA if b == 'categorical']
        X = pd.get_dummies(X, prefix=cats, prefix_sep='_', columns=cats)
        return X


class PriceModel(object):
    def __init__(self, X, y):
        self.model = LinearRegression(
            fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
        self.X = X
        self.y = y
        self.seed = 1

    def cv_predict(self):
        folds = cross_validation.KFold(
            len(self.y), n_folds=5, shuffle=True, random_state=self.seed)

        predictions = np.zeros_like(self.y)
        for train_i, test_i in folds:
            X_train, X_test = self.X[train_i], self.X[test_i]
            y_train = self.y[train_i]

            self.model.fit(X_train, y_train)
            predictions[test_i] = self.model.predict(X_test)
        return predictions
