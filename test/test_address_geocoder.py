import unittest
import pandas as pd
from real_estate.address_geocoder import (
    StreetscopeGeocoder, ElasticsearchServer, StreetscopeServer)


class TestElasticsearchServer(unittest.TestCase):
    def test(self):
        pass


class TestStreetscopeServer(unittest.TestCase):
    def test(self):
        pass


class TestStreetscopeGeocoder(unittest.TestCase):
    def test(self):
        pass

    @classmethod
    def setUpClass(self):
        # self.start_servers()
        self.geocoder = StreetscopeGeocoder()

    @classmethod
    def tearDownClass(self):
        # self.stop_servers()
        pass

    TEST_DF = pd.DataFrame(
        # 'address_text', 'bathrooms', 'bedrooms', 'first_encounted',
        # 'garage_spaces', 'last_encounted', 'postcode', 'price_max', 'price_min',
        # 'property_type', 'sale_type', 'sequence_broken', 'state',
        # 'under_application', 'under_contract', 'house', 'house_number', 'road',
        # 'suburb', 'address_is_valid', 'state_from_parser',
        # 'postcode_from_parser'
        columns=['house', 'house_number', 'road', 'suburb', 'state', 'postcode'],
        data=[
            [None,        '1',       'mills place',   'west beach',  'wa', 6450],
            [None,   '3/ 127',    'william street',    'st albans', 'vic', 3021],
            [None,   'b302/7',         'porter st',         'ryde', 'nsw', 2112],
            [None,       '24', 'balswidden street', 'albany creek', 'qld', 4035],
            [None,   '12/111',       'west street',    'mount isa', 'qld', 4825],
            [None,        '1',        'bluff road',     'queenton', 'qld', 4820],
            [None, 'lot 1812', 'stockland newport',      'newport', 'qld', 4020],
            [None,       '45',     'sorell street',    'chudleigh', 'tas', 7304],
            [None,       '96',       'senate road',   'port pirie',  'sa', 5540],
            [None,       '31',     'durham street',    'southport', 'qld', 4215],
        ]
    )

    URLS = ['http://localhost:5000/geocode?query=' + x for x in (
        '1+mills+place,+west+beach,+wa+6450',
        '3/+127+william+street,+st+albans,+vic+3021',
        'b302/7+porter+st,+ryde,+nsw+2112',
        '24+balswidden+street,+albany+creek,+qld+4035',
        '12/111+west+street,+mount+isa,+qld+4825',
        '1+bluff+road,+queenton,+qld+4820',
        'lot+1812+stockland+newport,+newport,+qld+4020',
        '45+sorell+street,+chudleigh,+tas+7304',
        '96+senate+road,+port+pirie,+sa+5540',
        '31+durham+street,+southport,+qld+4215',
    )]

    EXPECTED_RESULTS = [
        (0, 1), (0, 1), (0, 1), (0, 1), (0, 1),
        (0, 1), (0, 1), (0, 1), (0, 1), (0, 1),
    ]

    def test_mk_url(self):
        for i, r in enumerate(self.TEST_DF.iterrows()):
            self.assertEqual(self.geocoder.mk_url(r[1].values), self.URLS[i])

    def test_geocoding(self):
        addresses = self.geocoder.geocode_addresses(self.TEST_DF)
        for i, r in enumerate(addresses.iterrows()):
            self.assertEqual((r['lat'], r['long']), self.EXPECTED_RESULTS[i])
