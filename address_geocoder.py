import time
import re
import requests
import subprocess
import numpy as np
import pandas as pd


def expontial_backoff(url, current_delay, max_delay):
    while True:
        try:
            response = requests.get(url)
        except IOError:
            pass
        else:
            return response

        if current_delay < max_delay:
            raise Exception('Too many retry attempts.')
        time.sleep(current_delay)
        current_delay *= 2


class SimpleSubprocess(object):
    def __init__(self):
        pass

    def start(self):
        self.proc = subprocess.Popen(
            self.start_command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        time.sleep(self.start_up_time)

    def stop(self):
        self.proc.terminate()
        time.sleep(self.shut_down_time)

    def ping(self):
        try:
            requests.get(self.ping_url)
        except IOError:
            return False
        else:
            return True


class ElasticsearchServer(SimpleSubprocess):
    def __init__(self):
        self.start_command = 'elasticsearch'
        self.ping_url = 'http://localhost:9200/addresses'
        self.start_up_time = 30
        self.shut_down_time = 10


class StreetscopeServer(SimpleSubprocess):
    def __init__(self):
        self.start_command = ('python3', '../../streetscope/streetscope/app.py')
        self.ping_url = 'http://localhost:5000/about'
        self.start_up_time = 3
        self.shut_down_time = 1


class Geocoder(object):
    def __init__(self, verbose):
        self.verbose = verbose


class StreetscopeGeocoder(Geocoder):
    """
    Geocode addresses using Streetscope.
    streetscope: https://github.com/codeforamerica/streetscope
    """

    INDICIES = [
        'house', 'house_number', 'road', 'suburb', 'state', 'postcode'
    ]
    RE_SPACE = re.compile(r'\s+')

    def geocode_addresses(self, data):
        coords = pd.DataFrame(
            columns=['latitude', 'longitude', 'geocoding_validation'],
            data=list(data[self.INDICIES].apply(self.geocode, raw=True, axis=1).values)
        )
        return coords

    def geocode(self, r):
        url = self.mk_url(r)
        result = StreetscopeGeocoder.request(url)
        coords = self.process_result(result, r)
        return coords

    def mk_url(self, r):
        return (
            'http://localhost:5000/geocode?query=' +
            self.RE_SPACE.sub('+', '+'.join(filter(None.__ne__, map(
                lambda x: None if x[0] is None else str(x[0]) + x[1],
                zip(r, ('/', '', ',', ',', '', ''))
            ))))
        )

    def request(url):
        return expontial_backoff(url, 0.1, 5).json()

    def process_result(self, result, row):
        if result['total'] == 0:
            if self.verbose:
                print('No results.')
            return None
        else:
            hits = result['hits']
            matching_hits = list(filter(
                lambda x: self.filter_matches(
                    x['_source']['NUMBER'], row['house_number']
                ),
                hits
            ))

            if len(matching_hits) == 0:
                others = ', '.join(x['_source']['NUMBER'] for x in hits)
                if self.verbose:
                    print('No matches: %s - %s' % (row['house_number'], others))
                return [np.NaN, np.NaN, False]

            hit = matching_hits[0]['_source']
            return [float(hit['X']), float(hit['Y']), self.check_hit(hit, row)]

    ST_NO_REGEX = re.compile(r'.*/+\s*([^/]*)\s*|\s*([^/]*)\s*')
    def filter_matches(self, a, b):
        match = self.ST_NO_REGEX.match(str(b).lower())
        return str(a).lower() == (match.group(1) or match.group(2))

    STATE_CONVERSION = {
        'queensland': 'qld',
        'new south wales': 'nsw',
        'australian capital territory': 'act',
        'victoria': 'vic',
        'south australia': 'sa',
        'tasmania': 'tas',
        'western australia': 'wa',
        'northern territory': 'nt',
    }

    def check_hit(self, source, row):
        tests = [
            ('STREET', 'road'), ('CITY', 'suburb'), ('REGION', 'state'),
            ('POSTCODE', 'postcode')
        ]
        for a, b in tests:
            x, y = str(source[a]).lower(), str(row[b]).lower()
            if (
                self.STATE_CONVERSION.get(x, x) !=
                self.STATE_CONVERSION.get(y, y)
            ):
                if self.verbose:
                    print(
                        'Match failed: %s != %s [%s --- %s]' %
                        (x, y, str(source['ADDRESS']), str(row.values))
                    )
                return False
            return True


    # Notes:
    # - start this code afresh because it will need to be fast and reliable;
    # - this will require some thorough tests of the results returned by
    #   streetscope to ensure we have the correct location.

# {
#     'total': 3096387, 'max_score': 31.961416,
#     'hits': [
#         {
#             '_score': 31.961416, '_type': 'address', '_id': '14190144',
#             '_source': {
#                 'Y': '-33.87358000',
#                 'POSTCODE': '6450',
#                 'ADDRESS': '1 MILLS PLACE, WEST BEACH, WA 6450',
#                 'ACCURACY': '2',
#                 'STREET': 'MILLS PLACE',
#                 'CITY': 'WEST BEACH',
#                 'X': '121.88093000',
#                 'NUMBER': '1',
#                 'REGION': 'WA',
#                 'UNIT': ''
#             },
#             '_index': 'addresses'
#         },
#         {'_score': 29.131023, '_type': 'address', '_id': '1399606', '_source': {'Y': '-33.87302885', 'POSTCODE': '6450', 'ADDRESS': '12 MILLS PLACE, WEST BEACH, WA 6450', 'ACCURACY': '2', 'STREET': 'MILLS PLACE', 'CITY': 'WEST BEACH', 'X': '121.88020744', 'NUMBER': '12', 'REGION': 'WA', 'UNIT': ''}, '_index': 'addresses'},
#         {'_score': 29.116383, '_type': 'address', '_id': '12541702', '_source': {'Y': '-33.87379000', 'POSTCODE': '6450', 'ADDRESS': '7 MILLS PLACE, WEST BEACH, WA 6450', 'ACCURACY': '2', 'STREET': 'MILLS PLACE', 'CITY': 'WEST BEACH', 'X': '121.87985000', 'NUMBER': '7', 'REGION': 'WA', 'UNIT': ''}, '_index': 'addresses'},
#         {'_score': 29.101917, '_type': 'address', '_id': '1382993', '_source': {'Y': '-33.87357000', 'POSTCODE': '6450', 'ADDRESS': '4 MILLS PLACE, WEST BEACH, WA 6450', 'ACCURACY': '2', 'STREET': 'MILLS PLACE', 'CITY': 'WEST BEACH', 'X': '121.88044000', 'NUMBER': '4', 'REGION': 'WA', 'UNIT': ''}, '_index': 'addresses'},
#         {'_score': 29.101917, '_type': 'address', '_id': '6868527', '_source': {'Y': '-33.87329000', 'POSTCODE': '6450', 'ADDRESS': '9 MILLS PLACE, WEST BEACH, WA 6450', 'ACCURACY': '2', 'STREET': 'MILLS PLACE', 'CITY': 'WEST BEACH', 'X': '121.87978000', 'NUMBER': '9', 'REGION': 'WA', 'UNIT': ''}, '_index': 'addresses'},
#         {'_score': 29.101917, '_type': 'address', '_id': '9212178', '_source': {'Y': '-33.87302000', 'POSTCODE': '6450', 'ADDRESS': '10 MILLS PLACE, WEST BEACH, WA 6450', 'ACCURACY': '2', 'STREET': 'MILLS PLACE', 'CITY': 'WEST BEACH', 'X': '121.87979000', 'NUMBER': '10', 'REGION': 'WA', 'UNIT': ''}, '_index': 'addresses'},
#         {'_score': 29.101917, '_type': 'address', '_id': '11239951', '_source': {'Y': '-33.87382000', 'POSTCODE': '6450', 'ADDRESS': '6 MILLS PLACE, WEST BEACH, WA 6450', 'ACCURACY': '2', 'STREET': 'MILLS PLACE', 'CITY': 'WEST BEACH', 'X': '121.88006000', 'NUMBER': '6', 'REGION': 'WA', 'UNIT': ''}, '_index': 'addresses'},
#         {'_score': 29.09885, '_type': 'address', '_id': '3102971', '_source': {'Y': '-33.87335000', 'POSTCODE': '6450', 'ADDRESS': '3 MILLS PLACE, WEST BEACH, WA 6450', 'ACCURACY': '2', 'STREET': 'MILLS PLACE', 'CITY': 'WEST BEACH', 'X': '121.88063000', 'NUMBER': '3', 'REGION': 'WA', 'UNIT': ''}, '_index': 'addresses'},
#         {'_score': 29.09885, '_type': 'address', '_id': '3016846', '_source': {'Y': '-33.87308501', 'POSTCODE': '6450', 'ADDRESS': '2 MILLS PLACE, WEST BEACH, WA 6450', 'ACCURACY': '4', 'STREET': 'MILLS PLACE', 'CITY': 'WEST BEACH', 'X': '121.88077521', 'NUMBER': '2', 'REGION': 'WA', 'UNIT': ''}, '_index': 'addresses'},
#         {'_score': 29.09885, '_type': 'address', '_id': '9951118', '_source': {'Y': '-33.87260000', 'POSTCODE': '6450', 'ADDRESS': '15 MILLS PLACE, WEST BEACH, WA 6450', 'ACCURACY': '2', 'STREET': 'MILLS PLACE', 'CITY': 'WEST BEACH', 'X': '121.88002000', 'NUMBER': '15', 'REGION': 'WA', 'UNIT': ''}, '_index': 'addresses'}
#     ]
# }
