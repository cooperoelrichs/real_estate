from datetime import timedelta
import numpy as np
import pandas as pd


class DataValidationError(ValueError):
    pass

class ParallelListingsError(DataValidationError):
    pass

class UnbrokenListingsError(DataValidationError):
    pass

class UnorderedDateEncountedError(DataValidationError):
    pass

class UnorderedListingsError(DataValidationError):
    pass

class Unduplicator():
    '''Data Merger version 2.'''

    MAX_TIME_DIFF = timedelta(days=365/6)
    FIRST_ENCOUNTERED = 'first_encounted'
    LAST_ENCOUNTERED = 'last_encounted'
    ENCOUNTEREDS = [FIRST_ENCOUNTERED, LAST_ENCOUNTERED]

    NON_PROPERTY_COLUMNS = [
        'price_min', 'price_max', 'sequence_broken',
        FIRST_ENCOUNTERED, LAST_ENCOUNTERED
    ]

    def check_and_unduplicate(df):
        df = Unduplicator.sort_df_by_property_columns_and(
            df, Unduplicator.ENCOUNTEREDS)
        Unduplicator.check_ordering_of_listings(df)

        # Because we only partialy scrape states we can have unbrokens for
        # any date.
        # Unduplicator.check_for_unbrokens(df)

        Unduplicator.check_ordering_of_encounted_dates(df)
        df = Unduplicator.unduplicate(df)
        return df

    def unduplicate(df):
        df = df.copy()
        df['listings_subgroups'] = Unduplicator.make_subgroups_series(df)

        values = Unduplicator.get_le_and_sb_values(df)

        df = Unduplicator.sort_df_by_property_columns_and(
            df, ['listings_subgroups', Unduplicator.FIRST_ENCOUNTERED])
        dedup_filter = Unduplicator.dedup_filter_keeping_first(
            df, ['listings_subgroups'])

        filtered_df = df[dedup_filter].copy()
        filtered_df[Unduplicator.LAST_ENCOUNTERED] = values[Unduplicator.LAST_ENCOUNTERED]
        filtered_df['sequence_broken'] = values['sequence_broken']
        filtered_df['price_min'] = values['price_min']
        filtered_df['price_max'] = values['price_max']
        filtered_df = filtered_df.drop('listings_subgroups', axis=1)
        return filtered_df.sort_index()

    def get_le_and_sb_values(df):
        df = Unduplicator.sort_df_by_property_columns_and(
            df, ['listings_subgroups', Unduplicator.LAST_ENCOUNTERED])
        keep_last = Unduplicator.dedup_filter_keeping_last(
            df, ['listings_subgroups'])

        return {
            Unduplicator.LAST_ENCOUNTERED: df.loc[keep_last, Unduplicator.LAST_ENCOUNTERED].values,
            'sequence_broken': df.loc[keep_last, 'sequence_broken'].values,
            'price_min': df.loc[keep_last, 'price_min'].values,
            'price_max': df.loc[keep_last, 'price_max'].values
        }

    def dedup_filter_keeping_first(df, other_columns):
        return Unduplicator.dedup_filter(df, other_columns, 'first')

    def dedup_filter_keeping_last(df, other_columns):
        return Unduplicator.dedup_filter(df, other_columns, 'last')

    def dedup_filter(df, other_columns, keep):
        columns = Unduplicator.property_columns_plus(df, other_columns)
        return (~ df.duplicated(subset=columns, keep=keep))

    def property_columns_plus(df, other_columns):
        property_columns = Unduplicator.property_columns(df)
        return (list(property_columns) + other_columns)

    def make_subgroups_series(df):
        subgroups = ListingsSubgrouper.group(df)
        return subgroups

    def make_subgroups_series_using_the_old_method(df):
        # Create a new column that defines which 'group', where
        # a group is seperated on by 'gap', a property is in so that
        # the df can be sorted by property, 'group', first_encounted.
        property_columns = Unduplicator.property_columns(df)
        ls = ListingsSubgrouper(
            df, property_columns, Unduplicator.MAX_TIME_DIFF
        )
        sub_group_series = df.apply(ls.group, axis=1, raw=True, reduce=True)
        return sub_group_series

    def sort_df_by_property_columns_and(df, other_columns):
        columns = Unduplicator.property_columns_plus(df, other_columns)
        df = Unduplicator.sort_df_by(df, columns)
        return df

    def sort_df_by(df, by):
        return df.sort_values(by=by, inplace=False)

    def property_columns(df):
        return df.columns.difference(Unduplicator.NON_PROPERTY_COLUMNS)

    def check_ordering_of_listings(df):
        property_columns = Unduplicator.property_columns(df)

        equality_with_next = Unduplicator.equality_with_next(df)
        unorded_check = (
            (
                equality_with_next &
                (
                    df[Unduplicator.FIRST_ENCOUNTERED] >
                    df[Unduplicator.FIRST_ENCOUNTERED].shift(-1)
                )
            ) | (
                equality_with_next &
                (
                    df[Unduplicator.FIRST_ENCOUNTERED] ==
                    df[Unduplicator.FIRST_ENCOUNTERED].shift(-1)
                ) &
                (
                    df[Unduplicator.LAST_ENCOUNTERED] >
                    df[Unduplicator.LAST_ENCOUNTERED].shift(-1)
                )
            )
        )

        if unorded_check.any():
            raise UnorderedListingsError(
                '%i Unorded listings found.' % unorded_check.sum() +
                'Examples:\n%s' % str(
                    df[unorded_check | unorded_check.shift(1)]
                )
            )

    def equality_with_next(df):
        '''Assumes that the df is sorted by property_columns.'''
        property_columns = Unduplicator.property_columns(df)
        return df.duplicated(subset=property_columns, keep='last')

    def check_for_unbrokens(df):
        unbroken_last_encounteds = np.sort(
            df[~ df['sequence_broken']][Unduplicator.LAST_ENCOUNTERED].unique()
        )

        # unbroken_last_encounteds should unique to a single date.
        if unbroken_last_encounteds.shape[0] != 1:
            damaged_rows = df[
                df[Unduplicator.LAST_ENCOUNTERED].isin(
                    unbroken_last_encounteds[:-1]
                ) &
                ~ df['sequence_broken']
            ]

            raise UnbrokenListingsError(
                'There are unbroken records for multiple dates.\n' +
                'Broken rows:\n%s' % str(damaged_rows)
            )

    def check_ordering_of_encounted_dates(df):
        fe_gt_le = (
            df[Unduplicator.FIRST_ENCOUNTERED] > df[Unduplicator.LAST_ENCOUNTERED]
        )
        if fe_gt_le.any():
            raise UnorderedDateEncountedError(
                'Encounted dates failed the ordering check.\n' +
                'Rows that failed the check:\n%s' % str(df[fe_gt_le])
            )

class ListingsSubgrouper():
    def group(df):
        pcs = Unduplicator.property_columns(df)
        eq_prev = (df[pcs] == df[pcs].shift(1)).all(axis=1)
        time_diff = df['first_encounted'].subtract(df['last_encounted'].shift(1))
        gt_max = time_diff > Unduplicator.MAX_TIME_DIFF

        new_subgroup = ((~ eq_prev) | gt_max)
        subgroups = new_subgroup.cumsum()
        return subgroups


class OldListingsSubgrouper(object):
    def __init__(self, df, property_columns, max_time_diff):
        self.current_subgroup = None
        self.df = df
        self.df_len = len(df)
        self.property_columns = property_columns
        self.max_time_diff = max_time_diff

    def print_progress(self, i):
        f = 100
        if (i // (self.df_len/f) - (i-1) // (self.df_len/f)) == 1:
            print('            %i%%' % (i/self.df_len*100))

    def group(self, r):
        r_index = self.df.index.get_loc(r.name)
        previous_index = self.df.index[r_index - 1]

        self.print_progress(r_index)

        property_equal_to_previous = (
            r.name == 0 or
            not r[self.property_columns].equals(
                self.df.loc[previous_index, self.property_columns]
            )
        )

        if property_equal_to_previous:
            self.current_subgroup = 1
            return self.current_subgroup
        else:
            identicle_and_before_current = np.logical_and.reduce(
                np.array(list(
                    self.df[i] == x for (i, x) in
                    self.df.loc[r.name, self.property_columns].iteritems()
                )),
                axis=0
            )
            identicle_and_before_current[r_index:] = False

            max_last_encounted = self.df.loc[
                identicle_and_before_current,
                Unduplicator.LAST_ENCOUNTERED
            ].max()

            time_difference = (
                r[Unduplicator.FIRST_ENCOUNTERED] - max_last_encounted
            )
            new_subgroup = time_difference > self.max_time_diff

            if new_subgroup:
                self.current_subgroup += 1
            return self.current_subgroup
