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

class ListingsOrderError(DataValidationError):
    pass

class Merger(object):
    # for both sales and rentals
    MAX_TIME_DIFF = timedelta(days=365/2)
    NON_PROPERTY_COLUMNS = [
        'price_min', 'price_max', 'sequence_broken',
        'first_encounted', 'last_encounted'
    ]

    def check_and_merge_on_price_changes(df):
        Merger.check_ordering_of_encounted_dates(df)
        Merger.check_ordering_of_listings(df)
        Merger.check_for_unbrokens(df)
        df = Merger.merge_duplicated_listings(df)
        return df

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

    def check_ordering_of_listings(df):
        property_columns = Merger.property_columns(df)

        equality_with_next = Merger.property_equality_with_next(
            df, property_columns)
        unorded_check = (
            (
                equality_with_next &
                (df['first_encounted'] > df['first_encounted'].shift(-1))
            ) | (
                equality_with_next &
                (df['first_encounted'] < df['first_encounted'].shift(-1)) &
                (df['last_encounted'] > df['last_encounted'].shift(-1))
            )
        )
        if unorded_check.any():
            raise ListingsOrderError(
                '%i Unorded listings found.' % unorded_check.sum() +
                'Examples:\n%s' % str(df[unorded_check])
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
            parallel_listings = df[parallel_listings_filter]

            raise ParallelListingsError(
                '%i Parallel listings were found.\n' % len(parallel_listings) +
                'For example:\n%s' % str(parallel_listings[:10])
            )

    def property_equality_with_next(df, property_columns):
        # return (
        #     df[property_columns] == df[property_columns].shift(-1)
        # ).all(axis=1)
        return df.duplicated(subset=property_columns, keep='last')

    def property_equality_with_previous(df, property_columns):
        return df.duplicated(subset=property_columns, keep='first')

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

    def merge_duplicated_listings(df):
        # Merge identicle properties that have been off the market for less
        # than the maximum specified time difference.
        property_columns = Merger.property_columns(df)
        df = Merger.sort_df(df)

        time_diff_to_next_lt_max, _ = (
            Merger.create_time_diff_filters(df)
        )

        equality_with_next = Merger.property_equality_with_next(
            df, property_columns
        )

        updated_df = df[
            ~ (
                equality_with_next &
                time_diff_to_next_lt_max
            )
        ].copy()

        updated_df = Merger.update_first_encounted_values(df, updated_df)
        updated_df = Merger.update_last_encounted_values(df, updated_df)
        updated_df = Merger.update_sequence_broken_values(df, updated_df)
        raise RuntimeError('The above three functions assume that there is no maximum date seperation!!!')
        return updated_df.sort_index()

    def update_sequence_broken_values(df, updated_df):
        last_sb_values = Merger.get_sequence_broken_values_in_duplicates(df)
        lasts_in_duplicates = Merger.get_lasts_in_duplicates_filter(df)

        print(lasts_in_duplicates[lasts_in_duplicates].index)
        print(last_sb_values)

        updated_df.loc[
            lasts_in_duplicates, 'sequence_broken'
        ] = last_sb_values
        return updated_df

    def update_first_encounted_values(df, updated_df):
        min_fe_values = Merger.get_min_first_encounted_values_in_duplicates(df)
        lasts_in_duplicates = Merger.get_lasts_in_duplicates_filter(df)

        print(lasts_in_duplicates[lasts_in_duplicates].index)
        print(min_fe_values)

        updated_df.loc[
            lasts_in_duplicates, 'first_encounted'
        ] = min_fe_values
        return updated_df

    def update_last_encounted_values(df, updated_df):
        max_le_values = Merger.get_max_last_encounted_values_in_duplicates(df)
        lasts_in_duplicates = Merger.get_lasts_in_duplicates_filter(df)

        print(lasts_in_duplicates[lasts_in_duplicates].index)
        print(max_le_values)

        updated_df.loc[
            lasts_in_duplicates, 'last_encounted'
        ] = max_le_values
        return updated_df

    def get_lasts_in_duplicates_filter(df):
        property_columns = Merger.property_columns(df)

        _, time_diff_to_previous_lt_max = (
            Merger.create_time_diff_filters(df)
        )

        equality_with_next = Merger.property_equality_with_next(
            df, property_columns
        )
        equality_with_previous = Merger.property_equality_with_previous(
            df, property_columns
        )

        lasts_in_duplicates = (
            equality_with_previous &
            (~ equality_with_next) &
            time_diff_to_previous_lt_max
        )
        return lasts_in_duplicates


    def get_sequence_broken_values_in_duplicates(df):
        property_columns = Merger.property_columns(df)

        sorted_df = df.sort_values(
            list(property_columns) + ['sequence_broken'], ascending=True)

        _, time_diff_to_previous_lt_max = (
            Merger.create_time_diff_filters(sorted_df)
        )

        equality_with_next = Merger.property_equality_with_next(
            sorted_df, property_columns
        )
        equality_with_previous = Merger.property_equality_with_previous(
            sorted_df, property_columns
        )

        # The last is the max
        lasts_in_duplicates = (
            equality_with_previous &
            (~ equality_with_next) &
            time_diff_to_previous_lt_max
        )
        print(lasts_in_duplicates[lasts_in_duplicates].index)
        last_sb_values = sorted_df[lasts_in_duplicates]['sequence_broken'].values
        return last_sb_values

    def get_min_first_encounted_values_in_duplicates(df):
        property_columns = Merger.property_columns(df)

        sorted_df = df.sort_values(
            list(property_columns) + ['first_encounted'], ascending=True)

        time_diff_to_next_lt_max, _ = (
            Merger.create_time_diff_filters(sorted_df)
        )

        equality_with_next = Merger.property_equality_with_next(
            sorted_df, property_columns
        )
        equality_with_previous = Merger.property_equality_with_previous(
            sorted_df, property_columns
        )

        # The first is the min
        firsts_in_duplicates = (
            (~ equality_with_previous) &
            equality_with_next &
            time_diff_to_next_lt_max
        )
        print(firsts_in_duplicates[firsts_in_duplicates].index)
        min_le_values = sorted_df[firsts_in_duplicates]['first_encounted'].values
        return min_le_values

    def get_max_last_encounted_values_in_duplicates(df):
        property_columns = Merger.property_columns(df)

        sorted_df = df.sort_values(
            list(property_columns) + ['last_encounted'], ascending=True)

        _, time_diff_to_previous_lt_max = (
            Merger.create_time_diff_filters(sorted_df)
        )

        equality_with_next = Merger.property_equality_with_next(
            sorted_df, property_columns
        )
        equality_with_previous = Merger.property_equality_with_previous(
            sorted_df, property_columns
        )

        # The last is the max
        lasts_in_duplicates = (
            equality_with_previous &
            (~ equality_with_next) &
            time_diff_to_previous_lt_max
        )
        print(lasts_in_duplicates[lasts_in_duplicates].index)
        max_le_values = sorted_df[lasts_in_duplicates]['last_encounted'].values
        return max_le_values


    # def update_first_encounted_values(
    #     df, updated_df,
    #     time_diff_to_next_lt_max, time_diff_to_previous_lt_max,
    #     equality_with_next, equality_with_previous
    # ):
    #     firsts_in_duplicates = (
    #         (~ equality_with_previous) &
    #         equality_with_next &
    #         time_diff_to_next_lt_max
    #     )
    #     lasts_in_duplicates = (
    #         equality_with_previous &
    #         (~ equality_with_next) &
    #         time_diff_to_previous_lt_max
    #     )
    #
    #     initial_first_encounted_values = df.loc[
    #         firsts_in_duplicates, 'first_encounted'
    #     ].copy().values
    #
    #     updated_df.loc[
    #         lasts_in_duplicates, 'first_encounted'
    #     ] = initial_first_encounted_values
    #
    #     return updated_df

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
