import unittest
from real_estate.website_scraper import WebsiteScraper
import real_estate.real_estate_property as rep


class TestWebsiteScraper(unittest.TestCase):
    def test_group_scrapings(self):
        tests = [
            rep.Property(
                rep.SaleType(None, None, None, None),
                rep.Details('', 1, 1, 1, None, None),
                rep.Address(None, None, None, None, None, None)
            ),
            rep.Property(
                rep.SaleTypeParseFailed(),
                rep.Details('', 1, 1, 1, None, None),
                rep.Address(None, None, None, None, None, None)
            ),
            rep.Property(
                rep.SaleType(None, None, None, None),
                rep.Details('', 1, 1, 1, None, None),
                rep.AddressParseFailed(None, None)
            ),
            rep.DataContentTypeNotSupported('')
        ]

        parsed, failed = WebsiteScraper.split_scrapings(tests)
        self.assertEqual(parsed, tests[:1])
        self.assertEqual(failed, tests[1:])
