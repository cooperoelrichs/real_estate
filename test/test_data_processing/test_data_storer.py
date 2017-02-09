import unittest
from datetime import datetime as dt
import pandas as pd
from real_estate.data_processing.data_storer import DataStorer


class TestDataStorer(unittest.TestCase):
    TEST_DATA_DIR = 'real_estate/test/test_data/'
    TEST_CURRENT_DATA_FILE = TEST_DATA_DIR + 'current_data_sales.h5'
    TEST_NEW_DATA_FILE = TEST_DATA_DIR + 'new_data_sales.h5'
    TEST_UPDATED_DATA_FILE = TEST_DATA_DIR + 'updated_data_sales.h5'

    NEW_DATA_COLUMN_NAMES = ['str', 'int', 'flt', 'date_scraped']
    EXISTING_DATA_COLUMN_NAMES = [
        'str', 'int', 'flt',
        'first_encounted','last_encounted', 'sequence_broken'
    ]
    CURRENT_DATA = pd.DataFrame(
        columns=EXISTING_DATA_COLUMN_NAMES,
        data=[
            ['a', 1, 1.1, dt(2016, 1, 1, 0, 0, 0), dt(2016, 1, 1, 0, 0, 0), False],
            ['b', 2, 1.1, dt(2016, 1, 1, 0, 0, 0), dt(2016, 1, 2, 0, 0, 0), True],
            ['c', 1, 1.2, dt(2016, 1, 1, 0, 0, 0), dt(2016, 1, 1, 0, 0, 0), False],
            ['d', 2, 1.1, dt(2016, 1, 1, 0, 0, 0), dt(2016, 1, 2, 0, 0, 0), True],
            ['e', 3, 1.3, dt(2016, 1, 1, 0, 0, 0), dt(2016, 1, 1, 0, 0, 0), True],
            ['e', 3, 1.3, dt(2016, 1, 2, 0, 0, 0), dt(2016, 1, 3, 0, 0, 0), False],
        ]
    )
    NEW_DATA = pd.DataFrame(
        columns=NEW_DATA_COLUMN_NAMES,
        data=[
            ['a', 1, 1.1, dt(2016, 1, 9, 0, 0, 0)],
            ['b', 2, 1.1, dt(2016, 1, 9, 0, 0, 0)],
            ['c', 1, 1.3, dt(2016, 1, 9, 0, 0, 0)],
            ['c', 2, 1.2, dt(2016, 1, 9, 0, 0, 0)],
            ['cc', 2, 1.2, dt(2016, 1, 9, 0, 0, 0)],
            ['d', 2, 1.1, dt(2016, 1, 9, 0, 0, 0)],
            ['e', 3, 1.3, dt(2016, 1, 9, 0, 0, 0)],
        ]
    )
    EXPECTED_AFTER_UPDATING_UNBROKEN_SEQUENCES = pd.DataFrame(
        columns=EXISTING_DATA_COLUMN_NAMES,
        data=[
            ['a', 1, 1.1, dt(2016, 1, 1, 0, 0, 0), dt(2016, 1, 9, 0, 0, 0), False],
            ['b', 2, 1.1, dt(2016, 1, 1, 0, 0, 0), dt(2016, 1, 2, 0, 0, 0), True],
            ['c', 1, 1.2, dt(2016, 1, 1, 0, 0, 0), dt(2016, 1, 1, 0, 0, 0), True],
            ['d', 2, 1.1, dt(2016, 1, 1, 0, 0, 0), dt(2016, 1, 2, 0, 0, 0), True],
            ['e', 3, 1.3, dt(2016, 1, 1, 0, 0, 0), dt(2016, 1, 1, 0, 0, 0), True],
            ['e', 3, 1.3, dt(2016, 1, 2, 0, 0, 0), dt(2016, 1, 9, 0, 0, 0), False],
        ]
    )
    EXPECTED_AFTER_ADD_NEW = pd.DataFrame(
        columns=EXISTING_DATA_COLUMN_NAMES,
        data=[
            ['a', 1, 1.1, dt(2016, 1, 1, 0, 0, 0), dt(2016, 1, 1, 0, 0, 0), False],
            ['b', 2, 1.1, dt(2016, 1, 1, 0, 0, 0), dt(2016, 1, 2, 0, 0, 0), True],
            ['c', 1, 1.2, dt(2016, 1, 1, 0, 0, 0), dt(2016, 1, 1, 0, 0, 0), False],
            ['d', 2, 1.1, dt(2016, 1, 1, 0, 0, 0), dt(2016, 1, 2, 0, 0, 0), True],
            ['e', 3, 1.3, dt(2016, 1, 1, 0, 0, 0), dt(2016, 1, 1, 0, 0, 0), True],
            ['e', 3, 1.3, dt(2016, 1, 2, 0, 0, 0), dt(2016, 1, 3, 0, 0, 0), False],
            ['b', 2, 1.1, dt(2016, 1, 9, 0, 0, 0), dt(2016, 1, 9, 0, 0, 0), False],
            ['c', 1, 1.3, dt(2016, 1, 9, 0, 0, 0), dt(2016, 1, 9, 0, 0, 0), False],
            ['c', 2, 1.2, dt(2016, 1, 9, 0, 0, 0), dt(2016, 1, 9, 0, 0, 0), False],
            ['cc', 2, 1.2, dt(2016, 1, 9, 0, 0, 0), dt(2016, 1, 9, 0, 0, 0), False],
            ['d', 2, 1.1, dt(2016, 1, 9, 0, 0, 0), dt(2016, 1, 9, 0, 0, 0), False],
        ]
    )
    UPDATED_DATA = pd.DataFrame(
        columns=EXISTING_DATA_COLUMN_NAMES,
        data=[
            ['a', 1, 1.1, dt(2016, 1, 1, 0, 0, 0), dt(2016, 1, 9, 0, 0, 0), False],
            ['b', 2, 1.1, dt(2016, 1, 1, 0, 0, 0), dt(2016, 1, 2, 0, 0, 0), True],
            ['c', 1, 1.2, dt(2016, 1, 1, 0, 0, 0), dt(2016, 1, 1, 0, 0, 0), True],
            ['d', 2, 1.1, dt(2016, 1, 1, 0, 0, 0), dt(2016, 1, 2, 0, 0, 0), True],
            ['e', 3, 1.3, dt(2016, 1, 1, 0, 0, 0), dt(2016, 1, 1, 0, 0, 0), True],
            ['e', 3, 1.3, dt(2016, 1, 2, 0, 0, 0), dt(2016, 1, 9, 0, 0, 0), False],
            ['b', 2, 1.1, dt(2016, 1, 9, 0, 0, 0), dt(2016, 1, 9, 0, 0, 0), False],
            ['c', 1, 1.3, dt(2016, 1, 9, 0, 0, 0), dt(2016, 1, 9, 0, 0, 0), False],
            ['c', 2, 1.2, dt(2016, 1, 9, 0, 0, 0), dt(2016, 1, 9, 0, 0, 0), False],
            ['cc', 2, 1.2, dt(2016, 1, 9, 0, 0, 0), dt(2016, 1, 9, 0, 0, 0), False],
            ['d', 2, 1.1, dt(2016, 1, 9, 0, 0, 0), dt(2016, 1, 9, 0, 0, 0), False],
        ]
    )

    def test_update_unbroken_sequences(self):
        self.assertTrue(
            DataStorer.update_unbroken_sequences(
                self.CURRENT_DATA.copy(), self.NEW_DATA.copy()
            ).equals(
                self.EXPECTED_AFTER_UPDATING_UNBROKEN_SEQUENCES
            )
        )

    def test_add_new(self):
        self.assertTrue(
            DataStorer.add_new(
                self.CURRENT_DATA.copy(), self.NEW_DATA.copy()
            ).equals(
                self.EXPECTED_AFTER_ADD_NEW
            )
        )

    def test_update_data_using_synthetic_data(self):
        self.assertTrue(
            DataStorer.update_data(
                self.CURRENT_DATA.copy(), self.NEW_DATA.copy()
            ).equals(
                self.UPDATED_DATA
            )
        )

    def test_repeatedly_update_data_using_synthetic_data(self):
        updated_data = DataStorer.update_data(
            self.CURRENT_DATA.copy(), self.NEW_DATA.copy())
        repeatedly_updated_data = DataStorer.update_data(
            updated_data, self.NEW_DATA.copy())

        self.assertTrue(repeatedly_updated_data.equals(self.UPDATED_DATA))
