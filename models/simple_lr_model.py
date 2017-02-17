import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation
from real_estate.models.unduplicator import Unduplicator


class XY(object):
    """Generate X and y tensors from the properties DataFrame."""

    X_DATA = [
        ('bedrooms', 'continuous'),
        ('bathrooms', 'continuous'),
        ('garage_spaces', 'continuous'),
        ('property_type', 'categorical'),
        ('suburb', 'categorical')
    ]

    def setup_self(self, df, exclude_suburb, perform_merges):
        self.exclude_suburb = exclude_suburb
        df = self.filter_data(df)

        if perform_merges:
            df = Unduplicator.check_and_merge_on_price_changes(df)

        self.y = self.make_y(df)
        self.X = self.make_x(df)

    def general_data_filter(self, df):
        return (
            (np.isfinite(df['price_min']) | np.isfinite(df['price_max'])) &
            pd.notnull(df['suburb']) &
            np.isfinite(df['bedrooms']) &
            np.isfinite(df['bathrooms']) &
            np.isfinite(df['garage_spaces'])
        )

    def make_y(self, df):
        return df[['price_min', 'price_max']].mean(axis=1)

    def make_x(self, df):
        individualised_x_data = self.X_DATA
        if self.exclude_suburb:
            individualised_x_data -= {('suburb', 'categorical')}


        X = df[[a for a, _ in individualised_x_data]].copy()
        cats = [a for a, b in individualised_x_data if b == 'categorical']

        # Drop the 'Not Specified' property_type so that we have
        # identifiable coefficients.
        X.loc[X['property_type']=='Not Specified', 'property_type'] = np.NaN

        if not self.exclude_suburb:
            littlest_suburb = X['suburb'].value_counts().sort_index().sort_values().index[0]
            X.loc[X['suburb']==littlest_suburb, 'suburb'] = np.NaN

        X = pd.get_dummies(
            X, prefix=cats, prefix_sep='_', columns=cats,
            drop_first=False, dummy_na=False
        )

        return X


class SalesXY(XY):
    def __init__(self, df, exclude_suburb=False, perform_merges=True):
        self.setup_self(df, exclude_suburb, perform_merges)

    def sale_type_data_filter(self, df):
        return (
            (df['sale_type'] == 'Private Treaty') |
            (df['sale_type'] == 'Maybe Private Treaty')
        )

    def filter_data(self, df):
        # Required data: sale_type, price, and suburb
        df = df[
            self.sale_type_data_filter(df) &
            self.general_data_filter(df)
        ]

        return df


class RentalsXY(XY):
    def __init__(self, df, exclude_suburb=False, perform_merges=True):
        self.setup_self(df, exclude_suburb, perform_merges)

    def rentals_data_filter(self, df):
        return (
            (df['sale_type'] == 'Rental')
        )

    def filter_data(self, df):
        # Required data: sale_type, price, and suburb
        df = df[
            self.rentals_data_filter(df) &
            self.general_data_filter(df)
        ]
        return df


class PriceModel(object):
    def __init__(self, X, y):
        self.model = LinearRegression(
            fit_intercept=True, normalize=True, copy_X=True, n_jobs=1
        )
        self.X = X
        self.y = y
        self.seed = 1

    def predict(self):
        self.model.fit(self.X, self.y)
        return self.model.predict(self.X)

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
