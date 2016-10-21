import unittest
from real_estate.address_parser import RealEstateAddressParser
from real_estate.real_estate_property import AddressParseFailed
from real_estate.json_load_and_dump import JSONLoadAndDump


class TestAddressParser(unittest.TestCase):
    TEST_ADDRESSES = 'real_estate/test/data/test_addresses.json'

    def open_test_data(self):
        return JSONLoadAndDump.load_from_file(self.TEST_ADDRESSES)

    def test_parse_and_validate_address(self):
        parser = RealEstateAddressParser()
        test_cases = self.open_test_data()
        tests = test_cases['valid_address_strings']
        expected_results = test_cases['parsed_results']
        expected_results = [[(a, b) for a, b in x] for x in expected_results]

        self.assertEqual(len(tests), len(expected_results))

        for test, expected in zip(tests, expected_results):
            parsed = parser.parse_and_validate_address(test)
            self.assertEqual(parsed, expected, '\nTest Str: %s' % test)

    def test_parse_invalid_addresses(self):
        parser = RealEstateAddressParser()
        test_cases = self.open_test_data()
        tests = test_cases['invalid_address_strings']

        for test in tests:
            parsed = parser.parse_and_validate_address(test)
            self.assertIs(type(parsed), AddressParseFailed)
