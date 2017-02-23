import pandas as pd
import numpy as np
from real_estate.models.unduplicator import Unduplicator


class XY(object):
    """Generate X and y tensors from the properties DataFrame."""

    X_SPEC = [
        ('bedrooms', 'categorical'),
        ('bathrooms', 'categorical'),
        ('garage_spaces', 'categorical'),
        ('property_type', 'categorical'),
        ('suburb', 'categorical')
    ]

    VALID_SUBURBS_LIST = None  # Make this

    def setup_self(self, df, exclude_suburb, perform_merges):
        self.exclude_suburb = exclude_suburb
        self.perform_merges = perform_merges
        df = self.filter_data(df)

        if self.perform_merges:
            df = Unduplicator.check_and_unduplicate(df)

        self.data = df

    def filter_data(self, df):
        df = self.invalid_data_filter(df)
        df = self.qc_data_filter(df)
        return df

    def generaly_invalid_data_filter(self, df):
        return (
            (np.isfinite(df['price_min']) | np.isfinite(df['price_max'])) &
            pd.notnull(df['suburb']) &
            np.isfinite(df['bedrooms']) &
            np.isfinite(df['bathrooms']) &
            np.isfinite(df['garage_spaces'])
        )

    def qc_data_filter(self, df):
        return df[
            self.specific_qc_data_filter(df) &
            self.general_qc_data_filter(df)
        ]

    def general_qc_data_filter(self, df):
        return (
            pd.Series(np.ones(df.shape[0], dtype=bool), index=df.index)
            # df['suburb'].isin(self.VALID_SUBURBS_LIST)
        )

    def report_on_data_qc(self, df, outputs_dir):
        price_filtered = df[~ self.price_qc_filter(df)]
        generaly_filtered = df[~ self.general_qc_data_filter(df)]
        with open(outputs_dir + 'qc_data_filter_report.txt', 'w') as f:
            f.write(
                'Price Filter - from %i to %i records:\n%s' % (
                    len(df), len(price_filtered), str(price_filtered))
            )

            f.write('\n')

            f.write(
                'General Filter - from %i to %i records:\n%s' % (
                    len(df), len(generaly_filtered), str(generaly_filtered))
            )

    def specific_qc_data_filter(self, df):
        return self.price_qc_filter(df)

    def y(self):
        return self.make_y(self.data)

    def y_cv(self, i):
        return self.make_y(self.data.loc[i])

    def X(self):
        return self.make_x(self.data)

    def X_cv(self, i):
        return self.make_x(self.data.loc[i])

    def make_y(self, df):
        return df[['price_min', 'price_max']].mean(axis=1)

    def make_x(self, df):
        individualised_x_data = self.X_SPEC
        if self.exclude_suburb:
            individualised_x_data -= {('suburb', 'categorical')}


        X = df[[a for a, _ in individualised_x_data]].copy()
        cats = [a for a, b in individualised_x_data if b == 'categorical']

        # Drop the 'Not Specified' property_type so that we have
        # identifiable coefficients.
        X.loc[X['property_type']=='Not Specified', 'property_type'] = np.NaN

        if not self.exclude_suburb:
            littlest_suburb = X[
                'suburb'
            ].value_counts(
            ).sort_index(
            ).sort_values(
            ).index[0]

            X.loc[X['suburb']==littlest_suburb, 'suburb'] = np.NaN

        # Bedrooms categorical test
        X.loc[X['bedrooms']==1, 'bedrooms'] = np.NaN
        X.loc[X['bedrooms'] > 6, 'bedrooms'] = 6

        X.loc[X['bathrooms']==1, 'bathrooms'] = np.NaN
        X.loc[X['bathrooms'] > 6, 'bathrooms'] = 6

        X.loc[X['garage_spaces']==1, 'garage_spaces'] = np.NaN
        X.loc[X['garage_spaces'] > 6, 'garage_spaces'] = 6

        X = pd.get_dummies(
            X, prefix=cats, prefix_sep='_', columns=cats,
            drop_first=False, dummy_na=False
        )

        return X


class SalesXY(XY):
    def __init__(self, df, exclude_suburb=False, perform_merges=True):
        self.setup_self(df, exclude_suburb, perform_merges)

    def invalid_data_filter(self, df):
        # Required data: sale_type, price, and suburb
        return df[
            self.invalid_sale_data_filter(df) &
            self.generaly_invalid_data_filter(df)
        ]

    def invalid_sale_data_filter(self, df):
        return (
            (df['sale_type'] == 'Private Treaty') |
            (df['sale_type'] == 'Maybe Private Treaty')
        )

    def price_qc_filter(self, df):
        return (
            (df['price_min'] > 2 * 10 ** 4) &
            (df['price_max'] > 2 * 10 ** 4)
        )


class RentalsXY(XY):
    def __init__(self, df, exclude_suburb=False, perform_merges=True):
        self.setup_self(df, exclude_suburb, perform_merges)

    def invalid_data_filter(self, df):
        # Required data: sale_type, price, and suburb
        return df[
            self.invalid_rental_data_filter(df) &
            self.generaly_invalid_data_filter(df)
        ]

    def invalid_rental_data_filter(self, df):
        return (
            (df['sale_type'] == 'Rental')
        )

    def price_qc_filter(self, df):
        return (
            (df['price_min'] > 2 * 10 ** 1) &
            (df['price_max'] > 2 * 10 ** 1)
        )
