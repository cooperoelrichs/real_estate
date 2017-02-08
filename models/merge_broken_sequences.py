from datetime import timedelta
import numpy as np
import pandas as pd

class DataValidationError(ValueError):
    pass

class ParallelListingsError(DataValidationError):
    pass

class UnbrokenListingsError(DataValidationError):
    pass

class DateEncountedError(DataValidationError):
    pass

class Merger(object):
    # for both sales and rentals
    MAX_TIME_DIFF = timedelta(days=30)
    NON_PROPERTY_COLUMNS = [
        'price_min', 'price_max', 'sequence_broken',
        'first_encounted', 'last_encounted'
    ]

    def data_validation(df):
        Merger.check_for_unbrokens(df)
        Merger.check_for_parrallel_listings(df)

    def check_for_unbrokens(df):
        unbroken_last_encountereds = np.sort(
            df[~ df['sequence_broken']]['last_encounted'].unique()
        )
        if len(unbroken_last_encountereds) != 1:
            damaged_rows = df[
                df['last_encounted'].isin(unbroken_last_encountereds[:-1]) &
                ~ df['sequence_broken']
            ]
            raise UnbrokenListingsError(
                'There are unbroken records for multiple dates.\n' +
                'Broken rows:\n%s' % str(damaged_rows)
            )

    def check_for_parrallel_listings(df):
        property_columns = Merger.property_columns(df)
        df = Merger.sort_df(df)

        property_equality_with_next = (
            df[property_columns] == df[property_columns].shift(-1)
        ).all(axis=1)
        parallel_with_next = (
            df['last_encounted'] >= df['first_encounted'].shift(-1)
        )
        parallel_with_next[-1:] = False
        parallel_listings_check = (
            property_equality_with_next & parallel_with_next
        )

        if (parallel_listings_check).any():
            parallel_listings_filter = (
                parallel_listings_check | parallel_listings_check.shift(1)
            )
            parallel_listings_check[0] = parallel_listings_check[0]
            parallel_listings = df[parallel_listings_filter]

            raise ParallelListingsError(
                'Parallel listings were found.\n' +
                'Parallel listings:\n%s' % str(parallel_listings)
            )

    def check_ordering_of_encounted_dates(df):
        fe_gt_le = df['first_encounted'] > df['last_encounted']
        if fe_gt_le.any():
            raise DateEncountedError(
                'Encounted dates failed the ordering check.\n' +
                'Rows that failed the check:\n%s' % str(df[fe_gt_le])
            )

    def property_columns(df):
        return df.columns.difference(Merger.NON_PROPERTY_COLUMNS)

    def sort_df(df):
        property_columns = Merger.property_columns(df)
        df = df.sort_values(
            by=list(property_columns) + ['first_encounted', 'last_encounted'],
            inplace=False
        )
        return df

    def merge_on_price_change(df):
        # 1. Get duplicates
        # 2. Sort by property columns then first_encountered and
        #    last_encountered
        # 3. Filter time diffs greater than MAX_TIME_DIFF
        # 4. Filter on broken sequences
        # 5. Update the first_encountered column
        # 5. Return the filtered df

        Merger.check_ordering_of_encounted_dates(df)
        Merger.check_for_unbrokens(df)
        Merger.check_for_parrallel_listings(df)

        df = Merger.sort_df(df)

        property_columns = Merger.property_columns(df)
        time_diff = df['first_encounted'].shift(-1) - df['last_encounted']
        last_only = df.duplicated(subset=property_columns, keep='last')
        first_only = ~ df.duplicated(subset=property_columns, keep='first')

        # print(df[df.columns.difference(['sequence_broken'])])
        # print(first_only)
        # print(df[first_only][df.columns.difference(['sequence_broken'])])

        df_new = df[
            ~ df['sequence_broken'] |
            ~ last_only |
            ~ (
                (time_diff <= Merger.MAX_TIME_DIFF) &
                (time_diff >= timedelta(days=0))
            )
        ]

        # print(df_new[df.columns.difference(['sequence_broken'])])

        print(first_only.index)
        print(first_only)
        df_new.loc[:, 'first_encounted'] = df.loc[first_only, 'first_encounted']


        df_new = df_new.sort_index()
        print(df_new[df.columns.difference(['sequence_broken'])])
        return df_new
