import unittest
import pandas as pd
from real_estate.data_processing.data_storer import DataStorer


class TestUpdateDataUsingRealData(unittest.TestCase):
    TEST_DATA_DIR = 'real_estate/test/test_data/'
    TEST_CURRENT_DATA_FILE = TEST_DATA_DIR + 'current_data_sales.h5'
    TEST_NEW_DATA_FILE = TEST_DATA_DIR + 'new_data_sales.h5'
    TEST_UPDATED_DATA_FILE = TEST_DATA_DIR + 'updated_data_sales.h5'

    def update_data_using_real_data(self):
        # This test is slow, and so is excluded from the standard test suite.
        current_data = pd.read_hdf(self.TEST_CURRENT_DATA_FILE)
        new_data = pd.read_hdf(self.TEST_NEW_DATA_FILE)
        expected_updated_data = pd.read_hdf(self.TEST_UPDATED_DATA_FILE)

        resultant_updated_data = DataStorer.update_data(
            current_data.copy(), new_data.copy())

        self.assertEqual(
            expected_updated_data.shape, resultant_updated_data.shape)
        self.assertTrue(
            expected_updated_data.equals(resultant_updated_data))

        repeatedly_updated_data = DataStorer.update_data(
            resultant_updated_data.copy(), new_data.copy())

        self.assertEqual(
            expected_updated_data.shape, repeatedly_updated_data.shape)
        self.assertTrue(
            expected_updated_data.equals(repeatedly_updated_data))
