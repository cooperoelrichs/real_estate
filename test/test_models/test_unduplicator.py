import unittest
from datetime import datetime
from datetime import timedelta
import pandas as pd
import numpy as np
from pandas.util.testing import (assert_frame_equal, assert_series_equal)
from real_estate.models.unduplicator import (
    Unduplicator, ListingsSubgrouper,
    ParallelListingsError, UnbrokenListingsError,
    UnorderedDateEncountedError, UnorderedListingsError
)


class TestListingsSubgrouper(unittest.TestCase):
    COLUMNS = [
        'a', 'b', 'price_min', 'price_max',
        'first_encounted', 'last_encounted'
    ]

    UNMERGED = pd.DataFrame(
        data=[
            # 1. Normal merge.
            [1, 'a', 1, 2, datetime(2017, 1, 1), datetime(2017, 2, 1)],
            [1, 'a', 1, 2, datetime(2017, 2, 1), datetime(2017, 3, 1)],
            [1, 'a', 1, 2, datetime(2017, 3, 15), datetime(2017, 4, 1)],

            # 2. Normal don't merge
            [2, 'a', 1, 2, datetime(2017, 1, 1), datetime(2017, 2, 1)],
            [2, 'b', 1, 2, datetime(2017, 2, 1), datetime(2017, 3, 1)],
            [2, 'c', 1, 2, datetime(2017, 5, 1), datetime(2017, 6, 1)],

            # 3. Large date gap 1.
            [3, 'a', 1, 2, datetime(2017, 1, 1), datetime(2017, 2, 1)],
            [3, 'a', 1, 2, datetime(2017, 9, 1), datetime(2017, 10, 1)],
            [3, 'a', 1, 2, datetime(2017, 11, 1), datetime(2017, 12, 1)],

            # 4. Large date gap 1, extended test.
            [3, 'a', 1, 2, datetime(2018, 1, 1), datetime(2018, 2, 1)],
            [3, 'a', 1, 2, datetime(2018, 2, 15), datetime(2018, 3, 1)],
            [3, 'a', 1, 2, datetime(2018, 4, 1), datetime(2018, 4, 15)],

            # 5. Large date gap 2.
            [3, 'b', 1, 2, datetime(2017, 1, 1), datetime(2017, 2, 1)],
            [3, 'b', 1, 2, datetime(2017, 2, 1), datetime(2017, 3, 1)],
            [3, 'b', 1, 2, datetime(2017, 11, 1), datetime(2017, 12, 1)],

            # 6. Large date gap 3.
            [3, 'c', 1, 2, datetime(2017, 1, 1), datetime(2017, 2, 1)],
            [3, 'c', 1, 2, datetime(2017, 2, 1), datetime(2017, 3, 1)],
            [3, 'c', 1, 2, datetime(2018, 1, 1), datetime(2018, 2, 1)],

            # 7. Large date gap 4.
            [3, 'd', 1, 2, datetime(2017, 1, 1), datetime(2017, 2, 1)],
            [3, 'd', 1, 2, datetime(2017, 8, 1), datetime(2017, 12, 1)],
            [3, 'd', 1, 2, datetime(2018, 9, 1), datetime(2018, 10, 1)],

            # 8. Parallel 1.
            [4, 'a', 1, 2, datetime(2017, 1, 1), datetime(2017, 2, 1)],
            [4, 'a', 1, 2, datetime(2017, 2, 1), datetime(2017, 3, 1)],
            [4, 'a', 1, 2, datetime(2017, 1, 1), datetime(2017, 4, 1)],

            # 9. Parallel 2.
            [4, 'b', 1, 2, datetime(2017, 1, 1), datetime(2017, 5, 1)],
            [4, 'b', 1, 2, datetime(2017, 2, 1), datetime(2017, 3, 1)],
            [4, 'b', 1, 2, datetime(2017, 3, 1), datetime(2017, 4, 1)],

            # 10. Parallel with a large date gap.
            [4, 'c', 1, 2, datetime(2017, 1, 1), datetime(2017, 2, 1)],
            [4, 'c', 1, 2, datetime(2017, 9, 1), datetime(2017, 12, 1)],
            [4, 'c', 1, 2, datetime(2017, 11, 1), datetime(2017, 12, 1)],

            # 11. Price change
            [5, 'a', 1, 2, datetime(2017, 1, 1), datetime(2017, 2, 1)],
            [5, 'a', 3, 4, datetime(2017, 2, 1), datetime(2017, 3, 1)],
            [5, 'a', 4, 5, datetime(2017, 3, 15), datetime(2017, 4, 1)],
        ],
        columns=COLUMNS
    )

    SUBGROUPS = [
         1, 1, 1,  # 1
         2, 3, 4,  # 2
         5, 6, 6,  # 3
         6, 6, 6,  # 4
         7, 7, 8,  # 5
         9, 9,10,  # 6
        11,12,13,  # 7
        14,14,14,  # 8
        15,15,15,  # 9
        16,17,17,  # 10
        18,18,18,  # 11
    ]

    SUBGROUPS_ = [
        1,1,1,
        1,1,1,
        1,2,2,
        2,2,2,
        1,1,2,
        1,1,2,
        1,2,3,
        1,1,1,
        1,1,1,
        1,2,2,
        1,1,1,
    ]

    def test_group(self):
        # ls = ListingsSubgrouper(
        #     self.UNMERGED, Unduplicator.property_columns(self.UNMERGED),
        #     Unduplicator.MAX_TIME_DIFF
        # )
        #
        # for i, r in self.UNMERGED.iterrows():
        #     print(i, ls.group(r))
        #
        # resultant_subgroups = self.UNMERGED.apply(
        #     ls.group, axis=1, raw=True, reduce=True
        # ).values

        resultant_subgroups = ListingsSubgrouper.group(self.UNMERGED)
        np.testing.assert_array_equal(
            resultant_subgroups,
            self.SUBGROUPS
        )


class TestUnduplicator(unittest.TestCase):
    TEST_COLUMNS = [
        'a', 'b', 'sale_type', 'price_min', 'price_max',
        'first_encounted', 'last_encounted', 'sequence_broken'
    ]

    DF_WITH_PRICE_CHANGES = pd.DataFrame(
        data=[
            ['a', 1, 'private treaty', 1, 2, datetime(2017, 1, 1), datetime(2017, 2, 1), True],
            ['a', 1, 'private treaty', 3, 3, datetime(2017, 2, 2), datetime(2017, 3, 1), True],
            ['a', 1, 'auction', 3, 3, datetime(2017, 2, 2), datetime(2017, 9, 1), False],
            ['b', 1, 'private treaty', 2, 2, datetime(2017, 1, 1), datetime(2017, 1, 2), True],
            ['b', 1, 'private treaty', 2, 2, datetime(2017, 8, 1), datetime(2017, 9, 1), False],
            ['c', 1, 'private treaty', 4, 4, datetime(2017, 1, 1), datetime(2017, 2, 1), True],
            ['c', 1, 'private treaty', 4, 4, datetime(2017, 2, 2), datetime(2017, 4, 1), True],
            ['c', 1, 'private treaty', 5, 5, datetime(2017, 4, 2), datetime(2017, 9, 1), False],
        ],
        columns=TEST_COLUMNS
    )
    DF_WITH_PRICE_CHANGES_SUBGROUPS = pd.Series([1, 1, 2, 3, 4, 5, 5, 5])

    DF_WITH_MERGED_PRICE_CHANGES = pd.DataFrame(
        data=[
            ['a', 1, 'private treaty', 3, 3, datetime(2017, 1, 1), datetime(2017, 3, 1), True],
            ['a', 1, 'auction', 3, 3, datetime(2017, 2, 2), datetime(2017, 9, 1), False],
            ['b', 1, 'private treaty', 2, 2, datetime(2017, 1, 1), datetime(2017, 1, 2), True],
            ['b', 1, 'private treaty', 2, 2, datetime(2017, 8, 1), datetime(2017, 9, 1), False],
            ['c', 1, 'private treaty', 5, 5, datetime(2017, 1, 1), datetime(2017, 9, 1), False],
        ],
        columns=TEST_COLUMNS,
        index=[0, 2, 3, 4, 5]
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

    DF_WITH_PARALLELS = pd.DataFrame(
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

    DF_WITH_MERGED_PARALLELS = pd.DataFrame(
        data=[
            ['a', 1, 'x', 2, 3, datetime(2017, 1, 1), datetime(2017, 2, 1), True],
            ['a', 1, 'x', 2, 3, datetime(2017, 9, 1), datetime(2017, 9, 1), False],

            ['b', 1, 'x', 2, 3, datetime(2017, 1, 1), datetime(2017, 9, 1), False],
            ['c', 1, 'x', 2, 3, datetime(2017, 1, 1), datetime(2017, 3, 1), True],
            ['d', 1, 'x', 2, 3, datetime(2017, 1, 1), datetime(2017, 9, 1), False],
            ['e', 1, 'x', 2, 3, datetime(2017, 1, 1), datetime(2017, 9, 1), False],
        ],
        columns=TEST_COLUMNS,
        index=[0, 1, 2, 4, 6, 8]
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

    DF_WITH_ORDERED_LISTINGS = pd.DataFrame(
        data=[
            ['a', 1, 'x', 2, 3, datetime(2017, 1, 1), datetime(2017, 2, 1), True],
            ['a', 1, 'x', 2, 3, datetime(2017, 2, 1), datetime(2017, 3, 1), True],
            ['b', 1, 'x', 2, 3, datetime(2017, 2, 1), datetime(2017, 3, 1), True],
            ['b', 1, 'x', 2, 3, datetime(2017, 2, 1), datetime(2017, 3, 1), True],
            ['c', 1, 'x', 2, 3, datetime(2017, 1, 1), datetime(2017, 3, 1), True],
            ['c', 1, 'x', 2, 3, datetime(2017, 1, 2), datetime(2017, 2, 1), True],
        ],
        columns=TEST_COLUMNS
    )

    def test_make_subgroups_series(self):
        assert_series_equal(
            self.DF_WITH_PRICE_CHANGES_SUBGROUPS,
            Unduplicator.make_subgroups_series(self.DF_WITH_PRICE_CHANGES.copy())
        )

    def test_unduplicate_price_changes(self):
        assert_frame_equal(
            self.DF_WITH_MERGED_PRICE_CHANGES,
            Unduplicator.unduplicate(self.DF_WITH_PRICE_CHANGES.copy())
        )

    def test_unduplicate_parallels(self):
        assert_frame_equal(
            self.DF_WITH_MERGED_PARALLELS,
            Unduplicator.unduplicate(self.DF_WITH_PARALLELS.copy())
        )

    def test_check_for_unbrokens(self):
        self.assertRaises(
            UnbrokenListingsError,
            Unduplicator.check_for_unbrokens, self.DF_WITH_UNBROKENS.copy()
        )

    def test_check_ordering_of_listings(self):
        self.assertRaises(
            UnorderedListingsError,
            Unduplicator.check_ordering_of_listings,
            self.DF_WITH_UNORDERED_LISTINGS.copy()
        )

        try:
            Unduplicator.check_ordering_of_listings(
                self.DF_WITH_ORDERED_LISTINGS
            )
        except UnorderedListingsError:
            self.fail()

    def test_check_ordering_of_encounted_dates(self):
        self.assertRaises(
            UnorderedDateEncountedError,
            Unduplicator.check_ordering_of_encounted_dates,
            self.DF_WITH_FE_GT_LE.copy()
        )
