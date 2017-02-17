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

class ListingsSubgrouper(object):
    def __init__(self, df, property_columns, max_time_diff):
        self.current_subgroup = None
        self.df = df
        self.property_columns = property_columns
        self.max_time_diff = max_time_diff

    def group(self, r):
        previous_index = self.df.index[self.df.index.get_loc(r.name) - 1]

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
            identicle_properties = np.logical_and.reduce(
                np.array(list(
                    self.df[i] == x for (i, x) in
                    self.df.loc[r.name, self.property_columns].iteritems()
                )),
            axis=0)

            max_last_encounted = self.df.loc[
                identicle_properties,
                Unduplicator.LAST_ENCOUNTED
            ].loc[
                self.df.index[:self.df.index.get_loc(r.name)]
            ].max()
            time_difference = r[Unduplicator.FIRST_ENCOUNTED] - max_last_encounted
            new_subgroup = time_difference > self.max_time_diff

            if new_subgroup:
                self.current_subgroup += 1
            return self.current_subgroup


class Unduplicator():
    '''Data Merger version 2.'''

    MAX_TIME_DIFF = timedelta(days=365/2)
    FIRST_ENCOUNTED = 'first_encounted'
    LAST_ENCOUNTED = 'last_encounted'
    ENCOUNTEDS = [FIRST_ENCOUNTED, LAST_ENCOUNTED]

    NON_PROPERTY_COLUMNS = [
        'price_min', 'price_max', 'sequence_broken',
        FIRST_ENCOUNTED, LAST_ENCOUNTED
    ]

    def check_and_unduplicate(df):
        df = Unduplicator.sort_df_by_property_columns_and(
            df, Unduplicator.ENCOUNTEDS)
        Unduplicator.check_ordering_of_listings(df)
        Unduplicator.check_for_unbrokens(df)
        Unduplicator.check_ordering_of_encounted_dates(df)

        df = Unduplicator.unduplicate(df)
        return df

    def unduplicate(df):
        df = df.copy()
        df['listings_subgroups'] = Unduplicator.make_subgroups_series(df)

        values = Unduplicator.get_le_and_sb_values(df)

        df = Unduplicator.sort_df_by_property_columns_and(
            df, ['listings_subgroups', Unduplicator.FIRST_ENCOUNTED])
        dedup_filter = Unduplicator.dedup_filter_keeping_first(
            df, ['listings_subgroups'])

        filtered_df = df[dedup_filter].copy()
        filtered_df[Unduplicator.LAST_ENCOUNTED] = values[Unduplicator.LAST_ENCOUNTED]
        filtered_df['sequence_broken'] = values['sequence_broken']
        filtered_df['price_min'] = values['price_min']
        filtered_df['price_max'] = values['price_max']
        filtered_df = filtered_df.drop('listings_subgroups', axis=1)
        return filtered_df.sort_index()

    def get_le_and_sb_values(df):
        df = Unduplicator.sort_df_by_property_columns_and(
            df, ['listings_subgroups', Unduplicator.LAST_ENCOUNTED])
        keep_last = Unduplicator.dedup_filter_keeping_last(
            df, ['listings_subgroups'])

        return {
            Unduplicator.LAST_ENCOUNTED: df.loc[keep_last, Unduplicator.LAST_ENCOUNTED].values,
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
                    df[Unduplicator.FIRST_ENCOUNTED] >
                    df[Unduplicator.FIRST_ENCOUNTED].shift(-1)
                )
            ) | (
                equality_with_next &
                (
                    df[Unduplicator.FIRST_ENCOUNTED] ==
                    df[Unduplicator.FIRST_ENCOUNTED].shift(-1)
                ) &
                (
                    df[Unduplicator.LAST_ENCOUNTED] >
                    df[Unduplicator.LAST_ENCOUNTED].shift(-1)
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
            df[~ df['sequence_broken']][Unduplicator.LAST_ENCOUNTED].unique()
        )
        if len(unbroken_last_encounteds) != 1:
            damaged_rows = df[
                df[Unduplicator.LAST_ENCOUNTED].isin(unbroken_last_encounteds[:-1]) &
                ~ df['sequence_broken']
            ]
            raise UnbrokenListingsError(
                'There are unbroken records for multiple dates.\n' +
                'Broken rows:\n%s' % str(damaged_rows)
            )

    def check_ordering_of_encounted_dates(df):
        fe_gt_le = df[Unduplicator.FIRST_ENCOUNTED] > df[Unduplicator.LAST_ENCOUNTED]
        if fe_gt_le.any():
            raise UnorderedDateEncountedError(
                'Encounted dates failed the ordering check.\n' +
                'Rows that failed the check:\n%s' % str(df[fe_gt_le])
            )
