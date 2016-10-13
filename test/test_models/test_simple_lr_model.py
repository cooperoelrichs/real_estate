import unittest
import pandas as pd
import numpy as np
from real_estate.models.simple_lr_model import XY


class TestXY(unittest.TestCase):
    TEST_DF = pd.DataFrame(
        [
            ['Private Treaty', False, 2, 2, 'House', 2, 1, 1, 'a', 'ACT', 1],
            ['Private Treaty', False, 1, 2, 'Unit', 3, 2, 3, 'b', 'ACT', 2]
        ],
        columns=[
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
    )

    def test_xy(self):
        xy = XY(self.TEST_DF)
        self.assertEqual(xy.X.shape, (2, 7))
        self.assertEqual(xy.y.shape, (2,))
        self.assertIsInstance(xy.y, np.ndarray)
        np.testing.assert_array_equal(xy.y, np.array([2, 1.5]))
        self.assertEqual(set(xy.X_columns), {
            'bedrooms',
            'bathrooms',
            'garage_spaces',
            'property_type_House',
            'property_type_Unit',
            'suburb_a',
            'suburb_b'
        })
        np.testing.assert_array_equal(xy.X, np.array([
            [2, 3],
            [1, 2],
            [1, 3],
            [1, 0],
            [0, 1],
            [1, 0],
            [0, 1]
        ]).T)
