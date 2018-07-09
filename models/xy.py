import pandas as pd
import numpy as np
from sklearn import preprocessing
from real_estate.models.unduplicator import Unduplicator


class XY(object):
    """Generate X and y tensors from the properties DataFrame."""

    GENERIC_X_SPEC = [
        (('bedrooms',), 'continuous'),
        (('garage_spaces',), 'continuous'),
        (('bathrooms',), 'continuous'),

        # (('bedrooms',), 'polynomial'),
        # (('garage_spaces',), 'polynomial'),
        # (('bathrooms',), 'polynomial'),

        (('X',), 'continuous'),  # longitude
        (('Y',), 'continuous'),  # latitude
        # (('last_encounted',), 'time_sequence'),


        # (('bedrooms',), 'categorical'),
        # (('bathrooms',), 'categorical'),
        # (('garage_spaces',), 'categorical'),

        # (('bedrooms', 'property_type'), 'linear_by_categorical'),
        # (('bathrooms', 'property_type'), 'linear_by_categorical'),
        # (('garage_spaces', 'property_type'), 'linear_by_categorical'),

        # (('bathrooms', 'suburb'), 'linear_by_categorical'),
        # (('bedrooms', 'suburb'), 'linear_by_categorical'),
        # (('garage_spaces', 'suburb'), 'linear_by_categorical'),

        # (('property_type',), 'categorical'),
        # (('suburb',), 'categorical'),

        (('property_type',), 'numerically_encoded'),
        # (('suburb',), 'numerically_encoded'),
        # (('road',), 'numerically_encoded'),
    ]

    SALES_PRICE_LIMITS = (2e4, 1e7)
    RENTALS_PRICE_LIMITS = (2e1, 1e4)

    ORDINAL_EXCLUDE = 1
    ORDINAL_MAX = 8
    POLYNOMIAL_DEGREE = 3

    CATEGORICALS_EXCLUSIONS = {
        'property_type': 'Not Specified',
        'suburb': 'not specified',
        'bedrooms': 0,
        'bathrooms': 0,
        'garage_spaces': 0,
    }

    EXCLUDE_INVALID_GEOCODINGS = True
    MINIMUM_SUBURB_POPULATION = None
    EPOCH = np.datetime64('2017-11-09')

    DO_NAN_REPLACEMENTS = True
    NAN_REPLACEMENTS = ('bathrooms', 'bedrooms', 'garage_spaces')

    def setup_self(
        self, df, x_spec, perform_merges,
    ):
        self.x_spec = x_spec
        self.perform_merges = perform_merges
        df = self.filter_data(df)

        if self.perform_merges:
            df = Unduplicator.check_and_unduplicate(df)

        if self.DO_NAN_REPLACEMENTS:
            df = self.replace_nans(df)

        self.numerical_encoders = {}

        self.y = self.make_y(df)
        self.X = self.make_x(df)
        self.categorical_groups = self.make_categorical_groups(df)
        self.by_categorical_groups = self.make_by_categorical_groups(df)
        self.ne_groups = self.make_ne_groups(df)
        self.report_data_shape(
            self.X, self.y,
            self.categorical_groups, self.by_categorical_groups, self.ne_groups
        )

    def replace_nans(self, df):
        print('Performing nan replacment:')
        for c in self.NAN_REPLACEMENTS:
            print(' * %s - replacing %i nans' % (c, pd.isnull(df[c]).sum()))
            df.loc[pd.isnull(df[c]), c] = -1
        return df

    def report_data_shape(self, X, y, cats, by_cats, nes):
        n_cats = sum(x2-x1 for _, x1, x2 in cats)
        n_by_cats = sum(x2-x1 for _, _, x1, x2 in by_cats)
        print('Shape of X - %s' % str(X.shape))
        print('Shape of y - %s' % str(y.shape))
        print('X has %i categoricals,' % n_cats)
        print('X has %i by categoricals, and' % n_by_cats)
        print('X has %i numerically encoded features.' % len(nes))

    def filter_data(self, df):
        df = self.invalid_data_filter(df)
        df = self.qc_data_filter(df)

        if self.EXCLUDE_INVALID_GEOCODINGS:
            df = df[df['geocoding_is_valid']]
        if self.MINIMUM_SUBURB_POPULATION is not None:
            df = df[XY.minimum_suburb_population_filter(
                df, self.MINIMUM_SUBURB_POPULATION
            )]
        return df

    def experimental_generaly_invalid_data_filter(self, df):
        print('Warning, using the experimental invalid data filter.')
        return (np.isfinite(df['price_min']) | np.isfinite(df['price_max']))


    def generaly_invalid_data_filter(self, df):
        return (
            (np.isfinite(df['price_min']) | np.isfinite(df['price_max'])) &
            pd.notnull(df['suburb']) &
            np.isfinite(df['bedrooms']) &
            np.isfinite(df['bathrooms']) &
            np.isfinite(df['garage_spaces'])
        )

    def minimum_suburb_population_filter(df, min_count):
        n = df.shape[0]
        x = df['suburb'].groupby(df['suburb']).count().to_frame()
        x.columns=['suburb_count']
        f = df[['suburb']].join(other=x, on='suburb', how='left')
        f = f['suburb_count'] >= min_count

        x1 = len(df['suburb'].unique())
        x2 = len(df.loc[f, 'suburb'].unique())
        print('Using a minimum suburb population of %i records.' % min_count)
        print('Filtered from %i to %i (%.3f) records.' % (n, f.sum(), f.sum()/n))
        print('Leaving %i of %i (%.3f) unique suburbs.' % (x2, x1, x2/x1))
        return f

    def qc_data_filter(self, df):
        return df[
            self.specific_qc_data_filter(df) &
            self.general_qc_data_filter(df)
        ]

    def general_qc_data_filter(self, df):
        return pd.Series(np.ones(df.shape[0], dtype=bool), index=df.index)

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
        return self.x_spec

    def prep_work(self, df, x_spec):
        for categorical in [a for a, b in x_spec if b == 'categorical']:
            df = self.prep_categorical(categorical[0], df)
        for ordinal in [a for a, b in x_spec if b == 'ordinal']:
            df = self.prep_ordinal(ordinal[0], df)
        for polynomial in [a for a, b in x_spec if b == 'polynomial']:
            df = self.prep_polynomial(polynomial[0], df)
        for numerical in [a for a, b in x_spec if b == 'numerically_encoded']:
            df = self.prep_numerically_encoded(numerical[0], df)
        for time_sequence in [a for a, b in x_spec if b == 'time_sequence']:
            df = self.prep_time_sequence(time_sequence[0], df)

        set_of_linear_by_categorical = [
            a for a, b in x_spec if b == 'linear_by_categorical'
        ]

        for linear_by_categorical in set_of_linear_by_categorical:
            df = self.prep_linear_by_categorical(linear_by_categorical, df)
        return df

    def prep_time_sequence(self, feature, X):
        time_delta = pd.Series(
            data=(X[feature].values - self.EPOCH),
            index=X.index
        )
        X['last_encounted'] = time_delta.dt.days
        return X

    def prep_numerically_encoded(self, feature, X):
        X.loc[:, feature] = [str(a) for a in X[feature].values]
        le = preprocessing.LabelEncoder()
        le.fit(X[feature])
        X.loc[:, feature] = le.transform(X[feature])
        self.numerical_encoders[feature] = le
        return X

    def inverse_transform_encoded_feature(self, feature, X):
        return self.numerical_encoders[feature].inverse_transform(X[feature])


    def prep_categorical(self, categorical, X):
        # Drop the 'Not Specified' property_type so that we have
        # identifiable coefficients.
        X.loc[
            X[categorical] == self.CATEGORICALS_EXCLUSIONS[categorical],
            categorical
        ] = np.NaN
        return X

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

    def make_ne_groups(self, df):
        x_spec = self.get_individualised_x_spec()
        self.check_x_spec_ordering(x_spec)

        df = self.prep_work(df.copy(), x_spec)
        nes = [a[0] for a, b in x_spec if b == 'numerically_encoded']
        groups = [(feature, df.columns.get_loc(feature)) for feature in nes]
        return groups

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


class EmptyXY(XY):
    def __init__(self, X, y):
        self.X = X
        self.y = y


class SalesXY(XY):
    def __init__(self, df, x_spec, perform_merges=True):
        self.setup_self(df, x_spec, perform_merges)

    def invalid_data_filter(self, df):
        df = df[self.invalid_sale_data_filter(df)]
        df = df[self.experimental_generaly_invalid_data_filter(df)]
        return df

    def invalid_sale_data_filter(self, df):
        return (
            (df['sale_type'] == 'Private Treaty') |
            (df['sale_type'] == 'Maybe Private Treaty')
        )

    def price_qc_filter(self, df):
        return (
            (df['price_min'] > self.SALES_PRICE_LIMITS[0]) &
            (df['price_max'] > self.SALES_PRICE_LIMITS[0]) &
            (df['price_min'] < self.SALES_PRICE_LIMITS[1]) &
            (df['price_max'] < self.SALES_PRICE_LIMITS[1])
        )


class RentalsXY(XY):
    def __init__(self, df, x_spec, perform_merges=True):
        self.setup_self(df, x_spec, perform_merges)

    def invalid_data_filter(self, df):
        df = df[
            self.invalid_rental_data_filter(df) &
            self.experimental_generaly_invalid_data_filter(df)
        ]
        return df

    def invalid_rental_data_filter(self, df):
        return (
            (df['sale_type'] == 'Rental')
        )

    def price_qc_filter(self, df):
        return (
            (df['price_min'] > self.RENTALS_PRICE_LIMITS[0]) &
            (df['price_max'] > self.RENTALS_PRICE_LIMITS[0]) &
            (df['price_min'] < self.RENTALS_PRICE_LIMITS[1]) &
            (df['price_max'] < self.RENTALS_PRICE_LIMITS[1])
        )
