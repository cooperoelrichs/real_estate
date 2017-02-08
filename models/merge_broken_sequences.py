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
        unbroken_last_encounteds = np.sort(
            df[~ df['sequence_broken']]['last_encounted'].unique()
        )
        if len(unbroken_last_encounteds) != 1:
            damaged_rows = df[
                df['last_encounted'].isin(unbroken_last_encounteds[:-1]) &
                ~ df['sequence_broken']
            ]
            raise UnbrokenListingsError(
                'There are unbroken records for multiple dates.\n' +
                'Broken rows:\n%s' % str(damaged_rows)
            )

    def check_for_parrallel_listings(df):
        property_columns = Merger.property_columns(df)
        df = Merger.sort_df(df)

        equality_with_next = Merger.property_equality_with_next(
            df, property_columns)
        parallel_with_next = (
            df['last_encounted'] >= df['first_encounted'].shift(-1)
        )
        parallel_with_next[-1:] = False
        parallel_listings_check = (
            equality_with_next & parallel_with_next
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

    def property_equality_with_next(df, property_columns):
        # return (
        #     df[property_columns] == df[property_columns].shift(-1)
        # ).all(axis=1)
        return df.duplicated(subset=property_columns, keep='last')

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

    def check_and_merge_on_price_changes(df):
        Merger.check_ordering_of_encounted_dates(df)
        Merger.check_for_unbrokens(df)
        Merger.check_for_parrallel_listings(df)

        Merger.merge_on_price_changes(df)

    def merge_on_price_changes(df):
        # Merge identicle properties that have been off the market for less
        # than the maximum specified time difference.
        property_columns = Merger.property_columns(df)
        df = Merger.sort_df(df)

        time_diff_to_next_lt_max, time_diff_to_previous_lt_max = (
            Merger.create_time_diff_filters(df)
        )

        equality_with_next = Merger.property_equality_with_next(
            df, property_columns
        )
        equality_with_previous = df.duplicated(
            subset=property_columns, keep='first'
        )

        merged_df = df[
            ~ (
                equality_with_next &
                time_diff_to_next_lt_max
            )
        ].copy()

        updated_df = Merger.update_first_encounted_values(
            df, merged_df,
            time_diff_to_next_lt_max, time_diff_to_previous_lt_max,
            equality_with_next, equality_with_previous
        )
        return updated_df.sort_index()

    def update_first_encounted_values(
        df, merged_df,
        time_diff_to_next_lt_max, time_diff_to_previous_lt_max,
        equality_with_next, equality_with_previous
    ):
        firsts_in_duplicates = (
            (~ equality_with_previous) &
            equality_with_next &
            time_diff_to_next_lt_max
        )
        lasts_in_duplicates = (
            equality_with_previous &
            (~ equality_with_next) &
            time_diff_to_previous_lt_max
        )

        initial_first_encounted_values = df.loc[
            firsts_in_duplicates, 'first_encounted'
        ].copy().values

        merged_df.loc[
            lasts_in_duplicates, 'first_encounted'
        ] = initial_first_encounted_values

        return merged_df


    def create_time_diff_filters(df):
        time_diff_to_next = (
            df['first_encounted'].shift(-1) - df['last_encounted']
        )
        time_diff_to_next_lt_max = (time_diff_to_next <= Merger.MAX_TIME_DIFF)

        time_diff_to_previous = (
            df['first_encounted'] - df['last_encounted'].shift(1)
        )
        time_diff_to_previous_lt_max = (
            time_diff_to_previous <= Merger.MAX_TIME_DIFF
        )
        return time_diff_to_next_lt_max, time_diff_to_previous_lt_max
