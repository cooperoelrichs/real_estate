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
    EXPECTED_AFTER_UPDATE_LAST_SEEN = pd.DataFrame(
        columns=EXISTING_DATA_COLUMN_NAMES,
        data=[
            ['a', 1, 1.1, dt(1, 1, 1, 0, 0, 0), dt(1, 1, 9, 0, 0, 0), False],
            ['b', 2, 1.1, dt(1, 1, 1, 0, 0, 0), dt(1, 1, 9, 0, 0, 0), False],
            ['c', 1, 1.2, dt(1, 1, 1, 0, 0, 0), dt(1, 1, 1, 0, 0, 0), True],
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
            ['c', 1, 1.3, dt(1, 1, 9, 0, 0, 0), dt(1, 1, 9, 0, 0, 0), False],
            ['c', 2, 1.2, dt(1, 1, 9, 0, 0, 0), dt(1, 1, 9, 0, 0, 0), False],
            ['cc', 2, 1.2, dt(1, 1, 9, 0, 0, 0), dt(1, 1, 9, 0, 0, 0), False],
            ['d', 2, 1.1, dt(1, 1, 9, 0, 0, 0), dt(1, 1, 9, 0, 0, 0), False],
            ['e', 3, 1.3, dt(1, 1, 2, 0, 0, 0), dt(1, 1, 9, 0, 0, 0), False],
        ]
    )

    def test_update_data_store(self):
        self.assertEqual(True, False)

    def test_update_last_seen(self):
        self.assertEqual(True, False)

    def test_add_new(self):
        self.assertEqual(True, False)
