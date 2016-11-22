import unittest
from datetime import datetime as dt
import pandas as pd
from real_estate.data_storer import DataStorer


class TestDataStorer(unittest.TestCase):
    NEW_DATA_COLUMN_NAMES = ['str', 'int', 'flt', 'date_scraped']
    EXISTING_DATA_COLUMN_NAMES = [
        'str', 'int', 'flt',
        'first_encounted','last_encounted', 'sequence_broken'
    ]
    CURRENT_DATA = pd.DataFrame(
        columns=EXISTING_DATA_COLUMN_NAMES,
        data=[
            ['a', 1, 1.1, dt(1, 1, 1, 0, 0, 0), dt(1, 1, 1, 0, 0, 0), False],
            ['b', 2, 1.1, dt(1, 1, 1, 0, 0, 0), dt(1, 1, 2, 0, 0, 0), True],
            ['c', 1, 1.2, dt(1, 1, 1, 0, 0, 0), dt(1, 1, 1, 0, 0, 0), False],
            ['d', 2, 1.1, dt(1, 1, 1, 0, 0, 0), dt(1, 1, 2, 0, 0, 0), True],
            ['e', 3, 1.3, dt(1, 1, 1, 0, 0, 0), dt(1, 1, 1, 0, 0, 0), True],
            ['e', 3, 1.3, dt(1, 1, 2, 0, 0, 0), dt(1, 1, 3, 0, 0, 0), False],
        ]
    )
    NEW_DATA = pd.DataFrame(
        columns=NEW_DATA_COLUMN_NAMES,
        data=[
            ['a', 1, 1.1, dt(1, 1, 9, 0, 0, 0)],
            ['b', 2, 1.1, dt(1, 1, 9, 0, 0, 0)],
            ['c', 1, 1.3, dt(1, 1, 9, 0, 0, 0)],
            ['c', 2, 1.2, dt(1, 1, 9, 0, 0, 0)],
            ['cc', 2, 1.2, dt(1, 1, 9, 0, 0, 0)],
            ['d', 2, 1.1, dt(1, 1, 9, 0, 0, 0)],
            ['e', 3, 1.3, dt(1, 1, 9, 0, 0, 0)],
        ]
    )
    EXPECTED_AFTER_UPDATING_BROKEN_SEQUENCES = pd.DataFrame(
        columns=EXISTING_DATA_COLUMN_NAMES,
        data=[
            ['a', 1, 1.1, dt(1, 1, 1, 0, 0, 0), dt(1, 1, 1, 0, 0, 0), False],
            ['b', 2, 1.1, dt(1, 1, 1, 0, 0, 0), dt(1, 1, 2, 0, 0, 0), True],
            ['c', 1, 1.2, dt(1, 1, 1, 0, 0, 0), dt(1, 1, 1, 0, 0, 0), True],
            ['d', 2, 1.1, dt(1, 1, 1, 0, 0, 0), dt(1, 1, 2, 0, 0, 0), True],
            ['e', 3, 1.3, dt(1, 1, 1, 0, 0, 0), dt(1, 1, 1, 0, 0, 0), True],
            ['e', 3, 1.3, dt(1, 1, 2, 0, 0, 0), dt(1, 1, 3, 0, 0, 0), False],
        ]
    )
    EXPECTED_AFTER_UPDATING_LAST_ENCOUNTED = pd.DataFrame(
        columns=EXISTING_DATA_COLUMN_NAMES,
        data=[
            ['a', 1, 1.1, dt(1, 1, 1, 0, 0, 0), dt(1, 1, 9, 0, 0, 0), False],
            ['b', 2, 1.1, dt(1, 1, 1, 0, 0, 0), dt(1, 1, 9, 0, 0, 0), True],
            ['c', 1, 1.2, dt(1, 1, 1, 0, 0, 0), dt(1, 1, 1, 0, 0, 0), False],
            ['d', 2, 1.1, dt(1, 1, 1, 0, 0, 0), dt(1, 1, 2, 0, 0, 0), True],
            ['e', 3, 1.3, dt(1, 1, 1, 0, 0, 0), dt(1, 1, 1, 0, 0, 0), True],
            ['e', 3, 1.3, dt(1, 1, 2, 0, 0, 0), dt(1, 1, 9, 0, 0, 0), False],
        ]
    )
    EXPECTED_AFTER_ADD_NEW = pd.DataFrame(
        columns=EXISTING_DATA_COLUMN_NAMES,
        data=[
            ['a', 1, 1.1, dt(1, 1, 1, 0, 0, 0), dt(1, 1, 1, 0, 0, 0), False],
            ['b', 2, 1.1, dt(1, 1, 1, 0, 0, 0), dt(1, 1, 2, 0, 0, 0), True],
            ['c', 1, 1.2, dt(1, 1, 1, 0, 0, 0), dt(1, 1, 1, 0, 0, 0), False],
            ['d', 2, 1.1, dt(1, 1, 1, 0, 0, 0), dt(1, 1, 2, 0, 0, 0), True],
            ['e', 3, 1.3, dt(1, 1, 1, 0, 0, 0), dt(1, 1, 1, 0, 0, 0), True],
            ['e', 3, 1.3, dt(1, 1, 2, 0, 0, 0), dt(1, 1, 3, 0, 0, 0), False],
            ['b', 2, 1.1, dt(1, 1, 9, 0, 0, 0), dt(1, 1, 9, 0, 0, 0), False],
            ['c', 1, 1.3, dt(1, 1, 9, 0, 0, 0), dt(1, 1, 9, 0, 0, 0), False],
            ['c', 2, 1.2, dt(1, 1, 9, 0, 0, 0), dt(1, 1, 9, 0, 0, 0), False],
            ['cc', 2, 1.2, dt(1, 1, 9, 0, 0, 0), dt(1, 1, 9, 0, 0, 0), False],
            ['d', 2, 1.1, dt(1, 1, 9, 0, 0, 0), dt(1, 1, 9, 0, 0, 0), False],
            ['e', 3, 1.3, dt(1, 1, 2, 0, 0, 0), dt(1, 1, 9, 0, 0, 0), False],
        ]
    )
    UPDATED_DATA = pd.DataFrame(
        columns=EXISTING_DATA_COLUMN_NAMES,
        data=[
            ['a', 1, 1.1, dt(1, 1, 1, 0, 0, 0), dt(1, 1, 9, 0, 0, 0), False],
            ['b', 2, 1.1, dt(1, 1, 1, 0, 0, 0), dt(1, 1, 9, 0, 0, 0), True],
            ['c', 1, 1.2, dt(1, 1, 1, 0, 0, 0), dt(1, 1, 1, 0, 0, 0), True],
            ['d', 2, 1.1, dt(1, 1, 1, 0, 0, 0), dt(1, 1, 2, 0, 0, 0), True],
            ['e', 3, 1.3, dt(1, 1, 1, 0, 0, 0), dt(1, 1, 1, 0, 0, 0), True],
            ['e', 3, 1.3, dt(1, 1, 2, 0, 0, 0), dt(1, 1, 9, 0, 0, 0), False],
            ['c', 1, 1.3, dt(1, 1, 9, 0, 0, 0), dt(1, 1, 9, 0, 0, 0), False],
            ['c', 2, 1.2, dt(1, 1, 9, 0, 0, 0), dt(1, 1, 9, 0, 0, 0), False],
            ['cc', 2, 1.2, dt(1, 1, 9, 0, 0, 0), dt(1, 1, 9, 0, 0, 0), False],
            ['d', 2, 1.1, dt(1, 1, 9, 0, 0, 0), dt(1, 1, 9, 0, 0, 0), False],
            ['e', 3, 1.3, dt(1, 1, 2, 0, 0, 0), dt(1, 1, 9, 0, 0, 0), False],
        ]
    )

    def test_update_broken_sequences(self):
        self.assertTrue(
            DataStorer.update_broken_sequences(
                self.CURRENT_DATA, self.NEW_DATA
            ).equals(
                self.EXPECTED_AFTER_UPDATING_BROKEN_SEQUENCES
            )
        )

    def test_last_encounted(self):
        self.assertEqual(
            DataStorer.last_encounted(self.CURRENT_DATA, self.NEW_DATA),
            self.EXPECTED_AFTER_UPDATING_LAST_ENCOUNTED
        )

    def test_add_new(self):
        self.assertEqual(
            DataStorer.add_new(self.CURRENT_DATA, self.NEW_DATA),
            self.EXPECTED_AFTER_ADD_NEW
        )

    def test_update_data(self):
        self.assertEqual(
            DataStorer.update_data(self.CURRENT_DATA, self.NEW_DATA),
            self.UPDATED_DATA
        )
