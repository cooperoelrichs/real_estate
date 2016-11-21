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


class DataStorer():
    def update_data_store(new_data, file_path):
        current_data = pd.read_hdf(file_path)
        updated_data = DataStorer.update_last_seen(current_data, new_data)
        updated_data = DataStorer.add_new(updated_data, new_data)
        DataStorer.to_hdf(updated_data, file_path)

    def update_last_seen(current_data, new_data):
        pass


    def add_new(updated_data, new_data):
        pass


    def to_hdf(df, file_path):
        df.to_hdf(file_path, 'properties', append=False)
