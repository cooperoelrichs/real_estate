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

    ASSUMED_SS_LOCATION = '../../streetscope/streetscope/app.py'
    @classmethod
    def setUpClass(self):
        self.geocoder = StreetscopeGeocoder(False, self.ASSUMED_SS_LOCATION)
        self.geocoder.start_servers()

    @classmethod
    def tearDownClass(self):
        self.geocoder.stop_servers()
        pass

    TEST_DF = pd.DataFrame(
        columns=['house', 'house_number', 'road', 'suburb', 'state', 'postcode'],
        data=[
            ['%22"',          '1',       'mills place',   'west beach',  'wa', 6450],
            [  None,     '3/ 127',    'william street',    'st albans', 'vic', 3021],
            [  None,     'b302/7',         'porter st',         'ryde', 'nsw', 2112],
            [  None,      '24%22', 'balswidden street', 'albany creek', 'qld', 4035],
            [  None,     '12/111',       'west street',    'mount isa', 'qld', 4825],
            [  None,          '1',        'bluff road',     'queenton', 'qld', 4820],
            [  None,         '45',     'sorell street',    'chudleigh', 'tas', 7304],
            [  None,         '96',       'senate road',   'port pirie',  'sa', 5540],
            [  None,         '31',     'durham street',    'southport', 'qld', 4215],
            [  None,   'lot 1812', 'stockland newport',      'newport', 'qld', 4020],
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
        (-33.87358000, 121.88093000,  True),
        (-37.73708116, 144.80814973,  True),
        (-33.81962372, 151.09859390, False),
        (-27.34834231, 152.97401406,  True),
        (-20.72333991, 139.49313014, False),
        (-20.08156352, 146.26948710,  True),
        (-41.55761172, 146.48250547,  True),
        (-33.17437268, 137.99728970,  True),
        (-27.98985536, 153.41311390,  True),
        (      np.NaN,       np.NaN, False),
    ]


    def test_mk_url(self):
        data = self.geocoder.clean_strings(self.TEST_DF.copy())
        for i, r in enumerate(data.iterrows()):
            self.assertEqual(self.geocoder.mk_url(r[1].values), self.URLS[i])

    def test_geocoding(self):
        coords = self.geocoder.geocode_addresses(self.TEST_DF[:-1].copy())
        for i, index_row in enumerate(coords.iterrows()):
            _, r = index_row
            results = (r['latitude'], r['longitude'], r['geocoding_is_valid'])
            self.assertEqual(results, self.EXPECTED_RESULTS[i])

    def test_geocoding_failure(self):
        r = self.geocoder.geocode_addresses(self.TEST_DF[-1:].copy()).iloc[0]
        self.assertTrue(np.isnan(r['latitude']))
        self.assertTrue(np.isnan(r['longitude']))
        self.assertEqual(r['geocoding_is_valid'], self.EXPECTED_RESULTS[-1][2])
