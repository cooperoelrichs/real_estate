import unittest
import pandas as pd
import numpy as np
from real_estate.models.xy import (XY, SalesXY, RentalsXY)


class TestXY(unittest.TestCase):
    COLUMN_NAMES = [
        'sale_type',
        'under_contract',
        'price_min',
        'price_max',
        'property_type',
        'bedrooms',
        'bathrooms',
        'garage_spaces',
        'suburb',
        'state',
        'postcode'
    ]

    TEST_SALES_DF = pd.DataFrame(
        [
            ['Private Treaty', False, 1*10**6, 1*10**6, 'House', 2, 1, 1, 'a', 'ACT', 1],
            ['Private Treaty', False, 5*10**5, 2*10**6, 'Unit', 3, 2, 3, 'b', 'ACT', 2],
            ['Auction', False, 1*10**6, 1*10**6, 'Unit', 3, 2, 3, 'b', 'ACT', 2],
            ['Private Treaty', False, 100, 100, 'Unit', 3, 2, 3, 'b', 'ACT', 2],
        ],
        columns=COLUMN_NAMES
    )
    SALES_Y_VALUES = np.array([1*10**6, 1.25*10**6])

    TEST_RENTALS_DF = pd.DataFrame(
        [
            ['Rental', False, 100, 100, 'House', 2, 1, 1, 'a', 'ACT', 1],
            ['Rental', False, 50, 200, 'Unit', 3, 2, 3, 'b', 'ACT', 2],
            ['Negotiation', False, 100, 100, 'Unit', 3, 2, 3, 'b', 'ACT', 2],
            ['Rental', False, 10, 100, 'Unit', 3, 2, 3, 'b', 'ACT', 2],
        ],
        columns=COLUMN_NAMES
    )
    RENTALS_Y_VALUES = np.array([100, 125])

    def test_sales_xy(self):
        sales_xy = SalesXY(self.TEST_SALES_DF, perform_merges=False)
        np.testing.assert_array_equal(sales_xy.y, self.SALES_Y_VALUES)
        self.general_xy_tests(sales_xy)

    def test_rentals_xy(self):
        rentals_xy = RentalsXY(self.TEST_RENTALS_DF, perform_merges=False)
        np.testing.assert_array_equal(rentals_xy.y, self.RENTALS_Y_VALUES)
        self.general_xy_tests(rentals_xy)

    def general_xy_tests(self, xy):
        self.assertEqual(xy.X.shape, (2, 6))
        self.assertEqual(xy.y.shape, (2,))
        self.assertIsInstance(xy.X, pd.DataFrame)
        self.assertIsInstance(xy.X.values, np.ndarray)
        self.assertIsInstance(xy.y, pd.Series)
        self.assertIsInstance(xy.y.values, np.ndarray)
        self.assertEqual(list(xy.X.columns), [
            'bedrooms',
            'bathrooms',
            'garage_spaces',
            'property_type_House',
            'property_type_Unit',
            # 'suburb_a'  # Removed so that the parameters are identifiable.
            'suburb_b'
        ])

        np.testing.assert_array_equal(xy.X.values, np.array([
            [2, 3],
            [1, 2],
            [1, 3],
            [1, 0],
            [0, 1],
            [0, 1]
        ]).T)
