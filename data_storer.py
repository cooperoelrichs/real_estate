# 1. Recieve data.
# 2. Open data store.
# 3. Identify:
#  New data;
#  Existing data; and
#  Removed data.
# 4. Do:
#  Add new data;
#  Leave existing data; and
#  Record date of data removal.
# 5. Save updated data to store.

import pandas as pd
import numpy as np


class DataStorer():
    def update_data_store(new_data, file_path):
        current_data = pd.read_hdf(file_path)
        updated_data = DataStorer.update_data(current_data, new_data)
        DataStorer.to_hdf(updated_data, file_path)

    def update_data(current_data, new_data):
        wip_data = DataStorer.update_broken_sequences(current_data, new_data)
        wip_data = DataStorer.last_encounted(wip_data, new_data)
        updated_data = DataStorer.add_new(wip_data, new_data)
        return updated_data

    def update_broken_sequences(current_data, new_data):
        id_columns = list(new_data.columns.difference(['date_scraped']).values)

        x = current_data[current_data['sequence_broken'] == False]
        x = pd.merge(x, new_data, how='inner', on=id_columns)
        x = x[x['date_scraped'] > x['last_encounted']]

        current_data.loc[
            current_data['sequence_broken'] == False,
            'sequence_broken'
        ] = current_data.loc[
            current_data['sequence_broken'] == False,
            :
        ].apply(
            DataStorer.apply_sequence_break, axis=1, reduce=False, args=(
                x, list(current_data.columns.values)
            )
        )

        print(current_data)
        return current_data

    def apply_sequence_break(r, x, cols):
        return not np.all(list(x[i]==r[i] for i in cols), axis=0).any()

    def last_encounted(updated_data, new_data):
        # search:
        #  sequence_broken == False
        #  ignore first_encounted
        #  last_encounted < date_scraped
        #  identical match otherwise
        # do:
        #  update last_encounted
        #  update sequence_broken
        pass

    def add_new(updated_data, new_data):
        pass

    def to_hdf(df, file_path):
        df.to_hdf(file_path, 'properties', append=False)
