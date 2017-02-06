import pandas as pd
import numpy as np
import os
import datetime


class DataStorer():
    def create_new_unless_exists(df, file_path):
        if os.path.isfile(file_path):
            pass
        else:
            print('Data file did not exist, creating it: %s' % file_path)
            df = df.copy()
            df = DataStorer.reformat_new_data(df)
            DataStorer.to_hdf(df, file_path)

    def update_data_store(new_data, file_path):
        new_data = new_data.copy()
        current_data = pd.read_hdf(file_path)
        current_data = DataStorer.maybe_reformat_data(current_data)

        updated_data = DataStorer.update_data(current_data, new_data)
        DataStorer.to_hdf(updated_data, file_path)

    def update_data(current_data, new_data):
        wip_data = DataStorer.update_unbroken_sequences(current_data, new_data)
        updated_data = DataStorer.add_new(wip_data, new_data)
        return updated_data

    def update_unbroken_sequences(current_data, new_data):
        unbrokens = DataStorer.merge_unbrokens(current_data, new_data)
        updated_data = DataStorer.break_sequences(current_data, unbrokens)
        updated_data = DataStorer.update_last_encountered(
            updated_data, unbrokens)
        return updated_data

    def merge_unbrokens(current_data, new_data):
        id_columns = DataStorer.get_id_columns(new_data)

        unbrokens = current_data[
            DataStorer.sequence_unbroken_filter(current_data)
        ]
        unbrokens = pd.merge(unbrokens, new_data, how='inner', on=id_columns)
        unbrokens = unbrokens[
            unbrokens['date_scraped'] >= unbrokens['last_encounted']
        ]
        return unbrokens

    def sequence_unbroken_filter(data):
        return data['sequence_broken'] == False

    def eq_test(series, value):
        if isinstance(value, datetime.datetime):
            return series==value
        if isinstance(value, str):
            return series==value
        elif value is None:
            return series.isnull()
        elif np.isnan(value):
            return series.isnull()
        else:
            return series==value

    def break_sequences(current_data, unbrokens):
        current_data.loc[
            DataStorer.sequence_unbroken_filter(current_data),
            'sequence_broken'
        ] = current_data.loc[
            DataStorer.sequence_unbroken_filter(current_data),
            :
        ].apply(
            DataStorer.zero_matches, axis=1, reduce=False, args=(
                unbrokens, DataStorer.get_id_columns(current_data)
            )
        )
        return current_data

    def get_id_columns(df):
        return list(df.columns.difference(['date_scraped']).values)

    def zero_matches(r, x, cols):
        return not DataStorer.any_matches(r, x, cols)

    def any_matches(r, x, cols):
        return DataStorer.identicles_selection(r, x, cols).any()

    def identicles_selection(r, x, cols):
        return np.all(list(DataStorer.eq_test(x[i], r[i]) for i in cols), axis=0)

    def update_last_encountered(current_data, unbrokens):
        current_data.loc[
            DataStorer.sequence_unbroken_filter(current_data),
            'last_encounted'
        ] = current_data.loc[
            DataStorer.sequence_unbroken_filter(current_data),
            :
        ].apply(
            DataStorer.apply_last_enounted, axis=1, reduce=False, args=(
                unbrokens, DataStorer.get_id_columns(current_data)
            )
        )

        return current_data

    def apply_last_enounted(r, x, cols):
        return x.loc[
            DataStorer.identicles_selection(r, x, cols),
            'date_scraped'
        ].values[0]

    def add_new(current_data, new_data):
        id_columns = DataStorer.get_id_columns(new_data)

        unbrokens = current_data[
            DataStorer.sequence_unbroken_filter(current_data)
        ]

        new_uniques_filter = new_data.apply(
            DataStorer.zero_matches, axis=1, reduce=False, args=(
                unbrokens, id_columns
            )
        )
        new_uniques = new_data[new_uniques_filter].copy()

        if new_uniques.empty:
            return current_data
        else:
            new_uniques = DataStorer.reformat_new_data(new_uniques)
            updated_data = current_data.append(
                new_uniques, ignore_index=True, verify_integrity=True
            )
            return updated_data

    def maybe_reformat_data(df):
        if ('first_encounted' in df.columns and
                'last_encounted' in df.columns and
                'sequence_broken' in df.columns):
            return df
        elif (not 'first_encounted' in df.columns and
                not 'last_encounted' in df.columns and
                not 'sequence_broken' in df.columns and
                'date_scraped' in df.columns):
            return DataStorer.reformat_new_data(df)
        elif (not 'first_encounted' in df.columns and
                not 'last_encounted' in df.columns and
                not 'sequence_broken' in df.columns and
                'datetime' in df.columns):
            df.loc[:, 'date_scraped'] = df.loc[:, 'datetime']
            df = df.drop('datetime', 1)
            return DataStorer.reformat_new_data(df)
        else:
            raise RuntimeError('Invalid df: %s' % str(df.columns))

    def reformat_new_data(new_df):
        new_df.loc[:, 'first_encounted'] = new_df['date_scraped']
        new_df.loc[:, 'last_encounted'] = new_df['date_scraped']
        new_df.loc[:, 'sequence_broken'] = False
        new_df = new_df.drop('date_scraped', 1)
        return new_df

    def to_hdf(df, file_path):
        df.to_hdf(file_path, 'properties', append=False)
