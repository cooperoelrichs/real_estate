import pandas as pd
import numpy as np
import os
import datetime

from real_estate.memory_usage import MU


class DataStorer():
    NEW_COLUMNS = ('address_text',)

    def create_new_unless_exists(df, file_type, file_path):
        if os.path.isfile(file_path):
            pass
        else:
            print('Data file did not exist, creating it: %s' % file_path)
            df = df.copy()
            df = DataStorer.reformat_dataframe(df)
            DataStorer.to_ft(df, file_type, file_path)

    def update_data_store(new_data, file_type, file_path):
        new_shape = new_data.shape
        MU.print_memory_usage('07.01')
        # new_data = new_data.copy()
        MU.print_memory_usage('07.02')
        current_data = DataStorer.read_ft(file_type, file_path)
        current_shape = current_data.shape
        MU.print_memory_usage('07.03')
        current_data = DataStorer.maybe_apply_data_fixes(current_data)
        MU.print_memory_usage('07.04')
        updated_data = DataStorer.update_data(current_data, new_data)
        updated_shape = updated_data.shape
        MU.print_memory_usage('07.05')
        DataStorer.to_ft(updated_data, file_type, file_path)
        MU.print_memory_usage('07.06')

        print(
            'Data shape: new shape %s; current shape %s; updated shape %s.' %
            (str(new_shape), str(current_shape), str(updated_shape))
        )

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
        unbrokens = current_data[DataStorer.sequence_unbroken_filter(current_data)]
        unbrokens = pd.merge(unbrokens, new_data, how='inner', on=id_columns)
        unbrokens = unbrokens[unbrokens['date_scraped'] >= unbrokens['last_encounted']]
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
            new_uniques = DataStorer.reformat_dataframe(new_uniques)
            updated_data = current_data.append(
                new_uniques, ignore_index=True, verify_integrity=True
            )
            return updated_data

    def maybe_apply_data_fixes(df):
        df = DataStorer.maybe_add_missing_columns(df)
        df = DataStorer.maybe_reformat_data(df)
        return df

    def maybe_add_missing_columns(df):
        for x in DataStorer.NEW_COLUMNS:
            if x not in df.columns:
                df[x] = None
        return df

    def maybe_reformat_data(df):
        if ('first_encounted' in df.columns and
                'last_encounted' in df.columns and
                'sequence_broken' in df.columns):
            return df
        elif (not 'first_encounted' in df.columns and
                not 'last_encounted' in df.columns and
                not 'sequence_broken' in df.columns and
                'date_scraped' in df.columns):
            return DataStorer.reformat_dataframe(df)
        elif (not 'first_encounted' in df.columns and
                not 'last_encounted' in df.columns and
                not 'sequence_broken' in df.columns and
                'datetime' in df.columns):
            df.loc[:, 'date_scraped'] = df.loc[:, 'datetime']
            df = df.drop('datetime', 1)
            return DataStorer.reformat_dataframe(df)
        else:
            raise RuntimeError('Invalid df: %s' % str(df.columns))

    def reformat_dataframe(new_df):
        new_df.loc[:, 'first_encounted'] = new_df['date_scraped']
        new_df.loc[:, 'last_encounted'] = new_df['date_scraped']
        new_df.loc[:, 'sequence_broken'] = False
        new_df = new_df.drop('date_scraped', 1)
        return new_df

    def to_ft(df, file_type, file_path):
        if file_type == 'hdf':
            DataStorer.to_hdf(df, file_path)
        elif file_type == 'csv':
            DataStorer.to_csv(df, file_path)
        else:
            DataStorer.ft_error(file_type)

    def read_ft(file_type, file_path):
        if file_type == 'hdf':
            return pd.read_hdf(file_path)
        elif file_type == 'csv':
            return DataStorer.read_csv(file_path)
        else:
            DataStorer.ft_error(file_type)

    def ft_error(file_type):
        raise RuntimeError('File type not supported: %s.' % file_type)

    def to_hdf(df, file_path):
        df.to_hdf(file_path, 'properties', append=False)

    def to_csv(df, file_path):
        df.to_csv(file_path)

    def read_csv(file_path):
        df = pd.read_csv(file_path, index_col=0)

        for name in ('last_encounted', 'first_encounted'):
            df[name] = pd.to_datetime(
                df[name], format='%Y-%m-%d %H:%M:%S'
            )
        return df
