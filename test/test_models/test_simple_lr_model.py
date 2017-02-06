import unittest
import pandas as pd
import numpy as np
from real_estate.models.simple_lr_model import (XY, SalesXY, RentalsXY)


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
            ['Private Treaty', False, 2, 2, 'House', 2, 1, 1, 'a', 'ACT', 1],
            ['Private Treaty', False, 1, 2, 'Unit', 3, 2, 3, 'b', 'ACT', 2],
            ['Auction', False, 1, 2, 'Unit', 3, 2, 3, 'b', 'ACT', 2]
        ],
        columns=COLUMN_NAMES
    )

    TEST_RENTALS_DF = pd.DataFrame(
        [
            ['Rental', False, 2, 2, 'House', 2, 1, 1, 'a', 'ACT', 1],
            ['Rental', False, 1, 2, 'Unit', 3, 2, 3, 'b', 'ACT', 2],
            ['Negotiation', False, 1, 2, 'Unit', 3, 2, 3, 'b', 'ACT', 2]
        ],
        columns=COLUMN_NAMES
    )

    def test_sales_xy(self):
        (self.TEST_SALES_DF.columns)
        self.xy_tests(SalesXY(self.TEST_SALES_DF))

    def test_rentals_xy(self):
        self.xy_tests(RentalsXY(self.TEST_RENTALS_DF))

    def xy_tests(self, xy):
        self.assertEqual(xy.X.shape, (2, 6))
        self.assertEqual(xy.y.shape, (2,))
        self.assertIsInstance(xy.X, pd.DataFrame)
        self.assertIsInstance(xy.X.values, np.ndarray)
        self.assertIsInstance(xy.y, pd.Series)
        self.assertIsInstance(xy.y.values, np.ndarray)
        np.testing.assert_array_equal(xy.y, np.array([2, 1.5]))
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
