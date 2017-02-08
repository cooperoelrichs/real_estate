import unittest
from datetime import datetime
import pandas as pd
from pandas.util.testing import assert_frame_equal
from real_estate.models.merge_broken_sequences import (
    Merger, ParallelListingsError, UnbrokenListingsError, DateEncountedError
)


class TestMerger(unittest.TestCase):
    TEST_COLUMNS = [
        'a', 'b', 'sale_type', 'price_min', 'price_max',
        'first_encounted', 'last_encounted', 'sequence_broken'
    ]

    TEST_DF = pd.DataFrame(
        data=[
            ['a', 1, 'private treaty', 1, 2, datetime(2017, 1, 1), datetime(2017, 2, 1), True],
            ['a', 1, 'private treaty', 3, 3, datetime(2017, 2, 2), datetime(2017, 3, 1), True],
            ['a', 1, 'auction', 3, 3, datetime(2017, 2, 2), datetime(2017, 3, 1), False],
            ['b', 1, 'private treaty', 2, 2, datetime(2017, 4, 1), datetime(2017, 6, 1), False],
            ['c', 1, 'private treaty', 4, 4, datetime(2017, 1, 1), datetime(2017, 2, 1), True],
            ['c', 1, 'private treaty', 4, 4, datetime(2017, 2, 2), datetime(2017, 4, 1), True],
            ['c', 1, 'private treaty', 5, 5, datetime(2017, 4, 2), datetime(2017, 9, 1), False],
            ['b', 1, 'private treaty', 2, 2, datetime(2017, 1, 1), datetime(2017, 2, 1), True],
        ],
        columns=TEST_COLUMNS
    )

    EXPECTED_DF = pd.DataFrame(
        data=[
            ['a', 1, 'private treaty', 3, 3, datetime(2017, 1, 1), datetime(2017, 3, 1), True],
            ['a', 1, 'auction', 3, 3, datetime(2017, 2, 2), datetime(2017, 3, 1), False],
            ['b', 1, 'private treaty', 2, 2, datetime(2017, 4, 1), datetime(2017, 6, 1), False],
            ['c', 1, 'private treaty', 5, 5, datetime(2017, 1, 1), datetime(2017, 9, 1), False],
            ['b', 1, 'private treaty', 2, 2, datetime(2017, 1, 1), datetime(2017, 2, 1), True],
        ],
        columns=TEST_COLUMNS,
        index=[1, 2, 3, 6, 7]
    )

    DF_WITH_UNBROKENS = pd.DataFrame(
        data=[
            ['a', 1, 'x', 1, 2, datetime(2017, 1, 1), datetime(2017, 2, 1), False],
            ['a', 1, 'x', 3, 3, datetime(2017, 1, 1), datetime(2017, 3, 1), False],
            ['b', 1, 'x', 4, 4, datetime(2017, 1, 1), datetime(2017, 2, 1), True],
            ['b', 1, 'x', 4, 4, datetime(2017, 2, 2), datetime(2017, 3, 1), False],
        ],
        columns=TEST_COLUMNS
    )

    DF_WITH_FE_GT_LE = pd.DataFrame(
        data=[
            [datetime(2017, 2, 1), datetime(2017, 1, 1)],
        ],
        columns=['first_encounted', 'last_encounted']
    )

    DF_WITH_PARRALLELS = pd.DataFrame(
        data=[
            ['a', 1, 'x', 1, 2, datetime(2017, 1, 1), datetime(2017, 3, 1), True],
            ['a', 1, 'x', 3, 3, datetime(2017, 2, 1), datetime(2017, 3, 1), False],
            ['b', 1, 'x', 4, 4, datetime(2017, 1, 1), datetime(2017, 2, 1), True],
            ['b', 1, 'x', 4, 4, datetime(2017, 2, 2), datetime(2017, 3, 1), False],
        ],
        columns=TEST_COLUMNS
    )

    def test_merger(self):
        assert_frame_equal(
            self.EXPECTED_DF,
            Merger.merge_on_price_change(self.TEST_DF)
        )

    def test_check_for_unbrokens(self):
        self.assertRaises(
            UnbrokenListingsError,
            Merger.check_for_unbrokens, self.DF_WITH_UNBROKENS
        )

    def test_check_ordering_of_encounted_dates(self):
        self.assertRaises(
            DateEncountedError,
            Merger.check_ordering_of_encounted_dates, self.DF_WITH_FE_GT_LE
        )

    def test_check_for_parrallel_listings(self):
        self.assertRaises(
            ParallelListingsError,
            Merger.check_for_parrallel_listings, self.DF_WITH_PARRALLELS
        )
