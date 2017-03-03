import pandas as pd
import numpy as np
from real_estate.models.unduplicator import Unduplicator


class XY(object):
    """Generate X and y tensors from the properties DataFrame."""

    X_SPEC = [
        ('bedrooms', 'ordinal'),
        ('bathrooms', 'ordinal'),
        ('garage_spaces', 'ordinal'),
        ('property_type', 'categorical'),
        ('suburb', 'categorical')
    ]

    ORDINAL_EXCLUDE = 1
    ORDINAL_MAX = 6

    CATEGORICALS_EXCLUSIONS = {
        'property_type': 'Not Specified',
        'suburb': 'City'
    }
    EXCLUDE_SMALLEST_SUBURB = False

    VALID_SUBURBS_LIST = None  # Make this

    def setup_self(self, df, exclude_suburb, perform_merges):
        self.exclude_suburb = exclude_suburb
        self.perform_merges = perform_merges
        df = self.filter_data(df)

        if self.perform_merges:
            df = Unduplicator.check_and_unduplicate(df)

        self.y = self.make_y(df)
        self.X = self.make_x(df)

        self.dummy_groups = self.make_dummy_groups(df)

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

    def make_y(self, df):
        return df[['price_min', 'price_max']].mean(axis=1)

    def make_x(self, df):
        x_spec = self.get_individualised_x_spec()


        X = df[[a for a, _ in x_spec]].copy()
        cats = [a for a, b in x_spec if b != 'continuous']

        for categorical in [a for a, b in x_spec if b == 'categorical']:
            X = self.prep_categorical(categorical, X)
        for ordinal in [a for a, b in x_spec if b == 'ordinal']:
            X = self.prep_ordinal(ordinal, X)

        X = pd.get_dummies(
            X, prefix=cats, prefix_sep='_', columns=cats,
            drop_first=False, dummy_na=False
        )

        return X

    def get_individualised_x_spec(self):
        if self.exclude_suburb:
            return self.X_SPEC - {('suburb', 'categorical')}
        else:
            return self.X_SPEC

    def prep_categorical(self, categorical, X):
        # Drop the 'Not Specified' property_type so that we have
        # identifiable coefficients.
        X.loc[
            X[categorical] == self.CATEGORICALS_EXCLUSIONS[categorical],
            categorical
        ] = np.NaN

        if (
            categorical == 'suburb' and
            self.EXCLUDE_SMALLEST_SUBURB and
            not self.exclude_suburb
        ):
            X = self.exclude_smallest_suburb(X)
        return X

    def exclude_smallest_suburb(self, X):
        smallest_suburb = X[
            'suburb'
        ].value_counts(
        ).sort_index(
        ).sort_values(
        ).index[0]

        X.loc[X['suburb'] == smallest_suburb, 'suburb'] = np.NaN
        return X

    def prep_ordinal(self, ordinal, X):
        X.loc[X[ordinal] == self.ORDINAL_EXCLUDE, ordinal] = np.NaN
        X.loc[X[ordinal] > self.ORDINAL_MAX, ordinal] = self.ORDINAL_MAX
        return X

    def make_dummy_groups(self, df):
        x_spec = self.get_individualised_x_spec()
        contins = [a for a, b in x_spec if b == 'continuous']
        dummies = [(a, b) for a, b in x_spec if b != 'continuous']

        indicies = []
        current_pos = len(contins)

        for dummy, spec in dummies:
            uniques = df[dummy].unique()
            if spec == 'ordinal':
                uniques = uniques[
                    (uniques != self.ORDINAL_EXCLUDE) &
                    (uniques <= self.ORDINAL_MAX)
                ]
            if spec == 'categorical':
                uniques = uniques[
                    uniques != self.CATEGORICALS_EXCLUSIONS[dummy]
                ]

            num_uniques = uniques.shape[0]
            indicies += [
                (dummy, current_pos, current_pos + num_uniques)
            ]
            current_pos += num_uniques

        return indicies

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
