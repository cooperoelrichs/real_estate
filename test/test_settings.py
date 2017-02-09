import unittest
import shutil
import os
from real_estate.settings import Settings


class TestSettings(unittest.TestCase):
    TEST_DIR = os.path.join('real_estate', 'test', 'test_data')
    TEST_SETTINGS_FILE = os.path.join(TEST_DIR, 'test_settings.json')
    TEMP_TEST_DIR = os.path.join(TEST_DIR, 'settings_tests')

    EXPECTED_DATA_DIR = os.path.join(TEMP_TEST_DIR, 'data')
    EXPECTED_HTML_DIR = os.path.join(EXPECTED_DATA_DIR, 'html')

    EXPECTED_DATA_FILE = os.path.join(EXPECTED_DATA_DIR, 'data.h5')
    EXPECTED_HTML_FILE = os.path.join(EXPECTED_HTML_DIR, 'html.json')
    EXPECTED_FAILURES_LOG_FILE = os.path.join(EXPECTED_DATA_DIR, 'log.csv')

    def setUp(self):
        self.rm_test_dirs('set up')

    def tearDown(self):
        self.rm_test_dirs('tear down')

    def rm_test_dirs(self, stage_name):
        shutil.rmtree(self.TEMP_TEST_DIR, ignore_errors=True)

        if os.path.isdir(self.TEMP_TEST_DIR):
            raise RuntimeError(
                'Temp test dir still exists after %s - %s' %
                (stage_name, self.TEMP_TEST_DIR)
            )

    def test_settings(self):
        settings = Settings(
            self.TEST_SETTINGS_FILE, 'test_settings',
            self.TEMP_TEST_DIR, verbosity=False
        )

        self.assertIs(type(settings), Settings)
        self.assertEqual(settings.data_file, self.EXPECTED_DATA_FILE)
        self.assertEqual(settings.html_dump, self.EXPECTED_HTML_FILE)
        self.assertEqual(settings.failures_log,
                         self.EXPECTED_FAILURES_LOG_FILE)
        self.assertTrue(os.path.isdir(self.EXPECTED_DATA_DIR))
        self.assertTrue(os.path.isdir(self.EXPECTED_HTML_DIR))
