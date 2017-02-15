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
        identicle_properties = np.logical_and.reduce(
            np.array(list(
                self.df[i] == x for (i, x) in
                self.df.loc[r.name, self.property_columns].iteritems()
            )),
        axis=0)

        property_equal_to_previous = (
            r.name == 0 or
            not r[self.property_columns].equals(
                self.df.loc[r.name -1, self.property_columns]
            )
        )

        if property_equal_to_previous:
            self.current_subgroup = 1
            return self.current_subgroup
        else:
            max_last_encounted = self.df.loc[
                (self.df.index <= (r.name - 1)) & identicle_properties,
                'last_encounted'
            ].max()
            time_difference = r['first_encounted'] - max_last_encounted
            new_subgroup = time_difference > self.max_time_diff

            if new_subgroup:
                self.current_subgroup += 1
            return self.current_subgroup


class Unduplicator():
    '''Data Merger version 2.'''

    MAX_TIME_DIFF = timedelta(days=365/2)
    NON_PROPERTY_COLUMNS = [
        'price_min', 'price_max', 'sequence_broken',
        'first_encounted', 'last_encounted'
    ]

    def check_and_unduplicate(df):
        Unduplicator.check_ordering_of_listings(df)
        df = Unduplicator.unduplicate(df)
        return df

    def unduplicate(df):
        df = df.copy()
        df['listings_subgroups'] = Unduplicator.make_subgroups_series(df)

        values = Unduplicator.get_le_and_sb_values(df)

        df = Unduplicator.sort_df_property_columns_and(
            df, ['listings_subgroups', 'first_encounted'])
        dedup_filter = Unduplicator.dedup_filter_keeping_first(
            df, ['listings_subgroups'])

        filtered_df = df[dedup_filter].copy()
        filtered_df['last_encounted'] = values['last_encounted']
        filtered_df['sequence_broken'] = values['sequence_broken']
        filtered_df['price_min'] = values['price_min']
        filtered_df['price_max'] = values['price_max']
        filtered_df = filtered_df.drop('listings_subgroups', axis=1)
        return filtered_df.sort_index()

    def get_le_and_sb_values(df):
        df = Unduplicator.sort_df_property_columns_and(
            df, ['listings_subgroups', 'last_encounted'])
        keep_last = Unduplicator.dedup_filter_keeping_last(
            df, ['listings_subgroups'])

        return {
            'last_encounted': df.loc[keep_last, 'last_encounted'].values,
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
        return df.apply(ls.group, axis=1, raw=True, reduce=True)

    def sort_df_property_columns_and(df, other_columns):
        columns = Unduplicator.property_columns_plus(df, other_columns)
        df = Unduplicator.sort_df_by(df, columns)
        return df

    def sort_df_by(df, by):
        return df.sort_values(by=by, inplace=False)

    def property_columns(df):
        return df.columns.difference(Unduplicator.NON_PROPERTY_COLUMNS)

    def check_ordering_of_listings(df):
        property_columns = Unduplicator.property_columns(df)

        equality_with_next = Unduplicator.property_equality_with_next(
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
            raise UnorderedListingsError(
                '%i Unorded listings found.' % unorded_check.sum() +
                'Examples:\n%s' % str(df[unorded_check])
            )

    def equality_with_next(df):
        property_columns = Unduplicator.property_columns(df)
        return df.duplicated(subset=property_columns, keep='last')

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

    def check_ordering_of_encounted_dates(df):
        fe_gt_le = df['first_encounted'] > df['last_encounted']
        if fe_gt_le.any():
            raise UnorderedDateEncountedError(
                'Encounted dates failed the ordering check.\n' +
                'Rows that failed the check:\n%s' % str(df[fe_gt_le])
            )

    def check_ordering_of_listings(df):
        property_columns = Unduplicator.property_columns(df)

        equality_with_next = Unduplicator.equality_with_next(df)
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
            raise UnorderedListingsError(
                '%i Unorded listings found.' % unorded_check.sum() +
                'Examples:\n%s' % str(df[unorded_check])
            )
