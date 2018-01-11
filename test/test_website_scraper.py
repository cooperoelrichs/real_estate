import unittest
from real_estate.website_scraper import WebsiteScraper
import real_estate.real_estate_property as rep

from scraper.test.test_page_scraper import populate_state_and_postcode


class TestWebsiteScraper(unittest.TestCase):
    def test_group_scrapings(self):
        tests = [
            rep.Property(
                rep.SaleType(None, None, None, None),
                rep.Details('', 1, 1, 1, None, None),
                rep.AddressText(None)
            ),
            rep.Property(
                rep.SaleTypeParseFailed(),
                rep.Details('', 1, 1, 1, None, None),
                rep.AddressText(None)
            ),
            rep.DataContentTypeNotSupported('')
        ]
        tests = populate_state_and_postcode(tests, 'test_state', 9999)

        parsed, failed = WebsiteScraper.split_scrapings(tests)
        self.assertEqual(parsed, tests[:1])
        self.assertEqual(failed, tests[1:])
