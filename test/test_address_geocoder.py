import unittest
import numpy as np
import pandas as pd
from real_estate.address_geocoder import (
    StreetscopeGeocoder, ElasticsearchServer, StreetscopeServer)


class ServerTests():
    def start_server_test(test_class, server_object):
        server = server_object()
        server.start()
        test_class.assertTrue(server.ping())
        server.stop()

    def stop_server_test(test_class, server_object):
        server = server_object()
        server.start()
        server.stop()
        test_class.assertFalse(server.ping())


class TestElasticsearchServer(unittest.TestCase):
    def test_start_server(self):
        ServerTests.start_server_test(self, ElasticsearchServer)

    def test_stop_server(self):
        ServerTests.stop_server_test(self, ElasticsearchServer)


class TestStreetscopeServer(unittest.TestCase):
    def test_start_server(self):
        ServerTests.start_server_test(self, StreetscopeServer)

    def test_stop_server(self):
        ServerTests.stop_server_test(self, StreetscopeServer)


class TestStreetscopeGeocoder(unittest.TestCase):
    def test(self):
        pass

    @classmethod
    def setUpClass(self):
        self.ess = ElasticsearchServer()
        self.ess.start()
        self.sss = StreetscopeServer()
        self.sss.start()
        self.geocoder = StreetscopeGeocoder(False)

    @classmethod
    def tearDownClass(self):
        self.ess.stop()
        self.sss.stop()
        pass

    TEST_DF = pd.DataFrame(
        columns=['house', 'house_number', 'road', 'suburb', 'state', 'postcode'],
        data=[
            [None,        '1',       'mills place',   'west beach',  'wa', 6450],
            [None,   '3/ 127',    'william street',    'st albans', 'vic', 3021],
            [None,   'b302/7',         'porter st',         'ryde', 'nsw', 2112],
            [None,       '24', 'balswidden street', 'albany creek', 'qld', 4035],
            [None,   '12/111',       'west street',    'mount isa', 'qld', 4825],
            [None,        '1',        'bluff road',     'queenton', 'qld', 4820],
            [None,       '45',     'sorell street',    'chudleigh', 'tas', 7304],
            [None,       '96',       'senate road',   'port pirie',  'sa', 5540],
            [None,       '31',     'durham street',    'southport', 'qld', 4215],
            [None, 'lot 1812', 'stockland newport',      'newport', 'qld', 4020],
        ]
    )

    URLS = ['http://localhost:5000/geocode?query=' + x for x in (
        '1+mills+place,+west+beach,+wa+6450',
        '3/+127+william+street,+st+albans,+vic+3021',
        'b302/7+porter+st,+ryde,+nsw+2112',
        '24+balswidden+street,+albany+creek,+qld+4035',
        '12/111+west+street,+mount+isa,+qld+4825',
        '1+bluff+road,+queenton,+qld+4820',
        '45+sorell+street,+chudleigh,+tas+7304',
        '96+senate+road,+port+pirie,+sa+5540',
        '31+durham+street,+southport,+qld+4215',
        'lot+1812+stockland+newport,+newport,+qld+4020',
    )]

    EXPECTED_RESULTS = [
        (121.88093000, -33.87358000,  True),
        (144.80814973, -37.73708116,  True),
        (151.09859390, -33.81962372, False),
        (152.97401406, -27.34834231,  True),
        (139.49313014, -20.72333991, False),
        (146.26948710, -20.08156352,  True),
        (146.48250547, -41.55761172,  True),
        (137.99728970, -33.17437268,  True),
        (153.41311390, -27.98985536,  True),
        (      np.NaN,       np.NaN, False),
    ]


    def test_mk_url(self):
        for i, r in enumerate(self.TEST_DF.iterrows()):
            self.assertEqual(self.geocoder.mk_url(r[1].values), self.URLS[i])

    def test_geocoding(self):
        coords = self.geocoder.geocode_addresses(self.TEST_DF[:-1])
        for i, index_row in enumerate(coords.iterrows()):
            _, r = index_row
            results = (r['latitude'], r['longitude'], r['geocoding_validation'])
            self.assertEqual(results, self.EXPECTED_RESULTS[i])

    def test_geocoding_failure(self):
        r = self.geocoder.geocode_addresses(self.TEST_DF[-1:]).loc[0]
        self.assertTrue(np.isnan(r['latitude']))
        self.assertTrue(np.isnan(r['longitude']))
        self.assertFalse(r['geocoding_validation'])
