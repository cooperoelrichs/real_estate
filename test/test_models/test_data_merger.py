import unittest
from datetime import datetime
import pandas as pd
from pandas.util.testing import assert_frame_equal
from real_estate.models.data_merger import (
    Merger, ParallelListingsError, UnbrokenListingsError, DateEncountedError,
    ListingsOrderError
)


class TestMerger(unittest.TestCase):
    TEST_COLUMNS = [
        'a', 'b', 'sale_type', 'price_min', 'price_max',
        'first_encounted', 'last_encounted', 'sequence_broken'
    ]

    DF_WITH_PRICE_CHANGES = pd.DataFrame(
        data=[
            ['a', 1, 'private treaty', 1, 2, datetime(2017, 1, 1), datetime(2017, 2, 1), True],
            ['a', 1, 'private treaty', 3, 3, datetime(2017, 2, 2), datetime(2017, 3, 1), True],
            ['a', 1, 'auction', 3, 3, datetime(2017, 2, 2), datetime(2017, 9, 1), False],
            ['b', 1, 'private treaty', 2, 2, datetime(2017, 8, 1), datetime(2017, 9, 1), False],
            ['c', 1, 'private treaty', 4, 4, datetime(2017, 1, 1), datetime(2017, 2, 1), True],
            ['c', 1, 'private treaty', 4, 4, datetime(2017, 2, 2), datetime(2017, 4, 1), True],
            ['c', 1, 'private treaty', 5, 5, datetime(2017, 4, 2), datetime(2017, 9, 1), False],
            ['b', 1, 'private treaty', 2, 2, datetime(2017, 1, 1), datetime(2017, 1, 2), True],
        ],
        columns=TEST_COLUMNS
    )

    DF_WITH_MERGED_PRICE_CHANGES = pd.DataFrame(
        data=[
            ['a', 1, 'private treaty', 3, 3, datetime(2017, 1, 1), datetime(2017, 3, 1), True],
            ['a', 1, 'auction', 3, 3, datetime(2017, 2, 2), datetime(2017, 9, 1), False],
            ['b', 1, 'private treaty', 2, 2, datetime(2017, 8, 1), datetime(2017, 9, 1), False],
            ['c', 1, 'private treaty', 5, 5, datetime(2017, 1, 1), datetime(2017, 9, 1), False],
            ['b', 1, 'private treaty', 2, 2, datetime(2017, 1, 1), datetime(2017, 1, 2), True],
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
            ['a', 1, 'x', 2, 3, datetime(2017, 1, 1), datetime(2017, 2, 1), True],
            ['a', 1, 'x', 2, 3, datetime(2017, 9, 1), datetime(2017, 9, 1), False],

            ['b', 1, 'x', 2, 3, datetime(2017, 1, 1), datetime(2017, 9, 1), False],
            ['b', 1, 'x', 2, 3, datetime(2017, 2, 1), datetime(2017, 9, 1), False],

            ['c', 1, 'x', 2, 3, datetime(2017, 1, 1), datetime(2017, 2, 1), True],
            ['c', 1, 'x', 2, 3, datetime(2017, 1, 1), datetime(2017, 3, 1), True],

            ['d', 1, 'x', 2, 3, datetime(2017, 1, 1), datetime(2017, 9, 1), False],
            ['d', 1, 'x', 2, 3, datetime(2017, 1, 1), datetime(2017, 9, 1), False],

            ['e', 1, 'x', 2, 3, datetime(2017, 1, 1), datetime(2017, 9, 1), False],
            ['e', 1, 'x', 2, 3, datetime(2017, 2, 1), datetime(2017, 2, 2), True],
        ],
        columns=TEST_COLUMNS
    )

    DF_WITH_MERGED_PARRALLELS = pd.DataFrame(
        data=[
            ['a', 1, 'x', 2, 3, datetime(2017, 1, 1), datetime(2017, 2, 1), True],
            ['a', 1, 'x', 2, 3, datetime(2017, 9, 1), datetime(2017, 9, 1), False],

            ['b', 1, 'x', 2, 3, datetime(2017, 1, 1), datetime(2017, 9, 1), False],
            ['c', 1, 'x', 2, 3, datetime(2017, 1, 1), datetime(2017, 3, 1), True],
            ['d', 1, 'x', 2, 3, datetime(2017, 1, 1), datetime(2017, 9, 1), False],
            ['e', 1, 'x', 2, 3, datetime(2017, 1, 1), datetime(2017, 9, 1), False],
        ],
        columns=TEST_COLUMNS,
        index=[0, 1, 3, 5, 7, 9]
    )

    DF_WITH_UNORDERED_LISTINGS = pd.DataFrame(
        data=[
            ['a', 1, 'x', 2, 3, datetime(2017, 1, 1), datetime(2017, 2, 1), True],
            ['a', 1, 'x', 2, 3, datetime(2017, 2, 1), datetime(2017, 3, 1), True],
            ['b', 1, 'x', 2, 3, datetime(2017, 2, 1), datetime(2017, 3, 1), True],
            ['b', 1, 'x', 2, 3, datetime(2017, 1, 2), datetime(2017, 3, 1), True],
            ['c', 1, 'x', 2, 3, datetime(2017, 1, 1), datetime(2017, 3, 1), True],
            ['c', 1, 'x', 2, 3, datetime(2017, 1, 2), datetime(2017, 2, 1), True],
        ],
        columns=TEST_COLUMNS
    )

    # def test_merge_on_price_changes(self):
    #     assert_frame_equal(
    #         self.DF_WITH_MERGED_PRICE_CHANGES,
    #         Merger.merge_duplicated_listings(self.DF_WITH_PRICE_CHANGES)
    #     )

    # def test_merge_parallel_listings(self):
    #     assert_frame_equal(
    #         self.DF_WITH_MERGED_PARRALLELS,
    #         Merger.merge_duplicated_listings(self.DF_WITH_PARRALLELS)
    #     )

    def test_check_for_unbrokens(self):
        self.assertRaises(
            UnbrokenListingsError,
            Merger.check_for_unbrokens, self.DF_WITH_UNBROKENS
        )

    def test_check_ordering_of_listings(self):
        self.assertRaises(
            ListingsOrderError,
            Merger.check_ordering_of_listings, self.DF_WITH_UNORDERED_LISTINGS
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
