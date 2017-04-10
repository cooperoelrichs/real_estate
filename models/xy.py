import pandas as pd
import numpy as np
from real_estate.models.unduplicator import Unduplicator


class XY(object):
    """Generate X and y tensors from the properties DataFrame."""

    X_SPEC = [
        (('bedrooms',), 'polynomial'),
        (('garage_spaces',), 'polynomial'),
        (('bathrooms',), 'polynomial'),

        (('bedrooms', 'property_type'), 'linear_by_categorical'),
        (('bathrooms', 'property_type'), 'linear_by_categorical'),
        (('garage_spaces', 'property_type'), 'linear_by_categorical'),

        (('bathrooms', 'suburb'), 'linear_by_categorical'),
        (('bedrooms', 'suburb'), 'linear_by_categorical'),
        (('garage_spaces', 'suburb'), 'linear_by_categorical'),

        # (('bedrooms',), 'categorical'),
        # (('bathrooms',), 'categorical'),
        # (('garage_spaces',), 'categorical'),

        (('property_type',), 'categorical'),
        (('suburb',), 'categorical')
    ]

    ORDINAL_EXCLUDE = 1
    ORDINAL_MAX = 6
    POLYNOMIAL_DEGREE = 3

    CATEGORICALS_EXCLUSIONS = {
        'property_type': 'Not Specified',
        'suburb': 'not specified',  # 'city',
        'bedrooms': 0,
        'bathrooms': 0,
        'garage_spaces': 0,
    }
    # EXCLUDE_SMALLEST_SUBURB = False

    VALID_SUBURBS_LIST = None  # Make this

    def setup_self(self, df, exclude_suburb, perform_merges):
        self.exclude_suburb = exclude_suburb
        self.perform_merges = perform_merges
        df = self.filter_data(df)

        if self.perform_merges:
            df = Unduplicator.check_and_unduplicate(df)

        self.y = self.make_y(df)
        self.X = self.make_x(df)

        self.categorical_groups = self.make_categorical_groups(df)
        self.by_categorical_groups = self.make_by_categorical_groups(df)

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


        X = df[XY.reduce_tuples(
            [a for a, b in x_spec if b != 'linear_by_categorical']
        )].copy()
        cats = XY.reduce_tuples(
            [a for a, b in x_spec if b == 'categorical' or b == 'ordinal']
        )

        X = self.prep_work(X, x_spec)

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

    def prep_work(self, df, x_spec):
        for categorical in [a for a, b in x_spec if b == 'categorical']:
            df = self.prep_categorical(categorical[0], df)
        for ordinal in [a for a, b in x_spec if b == 'ordinal']:
            df = self.prep_ordinal(ordinal[0], df)
        for polynomial in [a for a, b in x_spec if b == 'polynomial']:
            df = self.prep_polynomial(polynomial[0], df)
        for linear_by_categorical in [a for a, b in x_spec
                                      if b == 'linear_by_categorical']:
            df = self.prep_linear_by_categorical(linear_by_categorical, df)
        return df

    def prep_categorical(self, categorical, X):
        # Drop the 'Not Specified' property_type so that we have
        # identifiable coefficients.
        X.loc[
            X[categorical] == self.CATEGORICALS_EXCLUSIONS[categorical],
            categorical
        ] = np.NaN

        # if (
        #     categorical == 'suburb' and
        #     self.EXCLUDE_SMALLEST_SUBURB and
        #     not self.exclude_suburb
        # ):
        #     X = self.exclude_smallest_suburb(X)
        return X

    # def exclude_smallest_suburb(self, X):
    #     smallest_suburb = X[
    #         'suburb'
    #     ].value_counts(
    #     ).sort_index(
    #     ).sort_values(
    #     ).index[0]
    #
    #     X.loc[X['suburb'] == smallest_suburb, 'suburb'] = np.NaN
    #     return X

    def prep_ordinal(self, ordinal, X):
        X.loc[X[ordinal] == self.ORDINAL_EXCLUDE, ordinal] = np.NaN
        X.loc[X[ordinal] > self.ORDINAL_MAX, ordinal] = self.ORDINAL_MAX
        return X

    def prep_polynomial(self, polynomial, X):
        if self.POLYNOMIAL_DEGREE != 3:
            raise ValueError('Only a POLYNOMIAL_DEGREE of 3 is supported.')

        X[polynomial + '_^2'] = X.loc[:, polynomial] ** 2
        X[polynomial + '_^3'] = X.loc[:, polynomial] ** 3
        return X

    def prep_linear_by_categorical(self, linear_by_categorical, X):
        linear, categorical = linear_by_categorical
        dummies = pd.get_dummies(
            X[[categorical]], prefix=linear,
            prefix_sep='_by_', columns=[categorical],
            drop_first=False, dummy_na=False
        )

        X[dummies.columns] = dummies.multiply(X[linear], axis=0)
        return X

    def make_categorical_groups(self, df):
        x_spec = self.get_individualised_x_spec()
        self.check_x_spec_ordering(x_spec)

        df = self.prep_work(df.copy(), x_spec)

        dummies = [
            (a[0], b) for a, b in x_spec
            if b == 'categorical' or b == 'ordinal'
        ]

        current_pos = self.calculate_position_of_first_dummy(x_spec, df)
        categorical_groups = []
        for dummy, _ in dummies:
            num_uniques = XY.num_uniques(df[dummy])
            categorical_groups += [
                (dummy, current_pos, current_pos + num_uniques)
            ]

            current_pos += num_uniques

        return categorical_groups

    def make_by_categorical_groups(self, df):
        x_spec = self.get_individualised_x_spec()
        self.check_x_spec_ordering(x_spec)

        contins = [a for a, b in x_spec if b == 'continuous']
        polynos = [a for a, b in x_spec if b == 'polynomial']
        pol_cat = [a for a, b in x_spec if b == 'linear_by_categorical']

        current_pos = len(contins) + len(polynos) * self.POLYNOMIAL_DEGREE

        df = self.prep_work(df.copy(), x_spec)
        groups = []
        for linear, categorical in pol_cat:
            num_uniques = XY.num_uniques(df[categorical])
            groups += [
                (linear, categorical, current_pos, current_pos + num_uniques)
            ]
            current_pos += num_uniques
        return groups

    def calculate_position_of_first_dummy(self, x_spec, df):
        contins = [a for a, b in x_spec if b == 'continuous']
        polynos = [a for a, b in x_spec if b == 'polynomial']
        pol_cat = [
            (a, b) for a, b in x_spec
            if b == 'linear_by_categorical'
        ]

        return (
            len(contins) +
            len(polynos) * self.POLYNOMIAL_DEGREE +
            sum([XY.num_uniques(df[a[1]]) for a, _ in pol_cat]) - 1
        )

    def num_uniques(series):
        return series.unique()[~pd.isnull(series.unique())].shape[0]

    def check_x_spec_ordering(self, x_spec):
        p = False
        for q in [b == 'categorical' for a, b in x_spec]:
            if p and p != q:
                raise RuntimeError('X_SPEC ordering is not correct')
            p = q

    def reduce_tuples(list_of_tuples):
        return list(sum(list_of_tuples, ()))


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
