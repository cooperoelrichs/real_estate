import time
import re
import requests
import subprocess
import numpy as np
import pandas as pd

import json


def expontial_backoff(url, current_delay, max_delay):
    while True:
        try:
            response = requests.get(url)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            if current_delay > max_delay:
                raise e
            time.sleep(current_delay)
            current_delay *= 2
        else:
            return response


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
        except requests.exceptions.RequestException as e:
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
    def __init__(self, app_location):
        self.start_command = ('python3', app_location)
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

    def __init__(self, verbose, streetscope_location):
        super().__init__(verbose)
        self.elasticsearch_server = ElasticsearchServer()
        self.streetscope_server = StreetscopeServer(streetscope_location)
        self.servers_running = False

    def start_servers(self):
        if self.verbose:
            print('Starting the elasticsearch server.')
        self.elasticsearch_server.start()

        if self.verbose:
            print('Starting the streetscope server.')
        self.streetscope_server.start()

        self.servers_running = True

    def check_servers(self):
        for name, server in (
            ('Elasticsearch', self.elasticsearch_server),
            ('Streetscope', self.streetscope_server),
        ):
            if not server.ping():
                raise RuntimeError("%s server didn't respond." % name)

    def stop_servers(self):
        self.elasticsearch_server.stop()
        self.streetscope_server.stop()
        self.servers_running = False

    def geocode_addresses(self, data):
        if not self.servers_running:
            raise RuntimeError('Geocoding servers have not been started.')
        self.start_time = time.time()
        self.data_len = len(data)

        data = self.clean_strings(data)

        column_names = ['latitude', 'longitude', 'geocoding_is_valid']
        coords = pd.DataFrame(
            columns=column_names,
            data=list(data[self.INDICIES].apply(
                self.geocode, raw=True, axis=1
            ).values),
            index=data.index
        )

        for a in column_names:
            data[a] = coords[a]
        return data

    def clean_strings(self, data):
        for a in ['house', 'house_number', 'road', 'suburb']:
            data[a] = data[a].map(self.clean_string)
        return data

    QUOTES_REGEX = re.compile(r'"|\'|\*|%22')
    def clean_string(self, x):
        if isinstance(x, str):
            x = self.QUOTES_REGEX.sub('', x)
            return x.strip()
        else:
            return x

    def geocode(self, r):
        if r.name % 1000 == 0:
            self.progress_summary(self.start_time, r.name, self.data_len)

        url = self.mk_url(r)
        result = StreetscopeGeocoder.request(url, r)
        coords = self.process_result(result, r)
        return coords

    def mk_url(self, r):
        return (
            'http://localhost:5000/geocode?query=' +
            self.RE_SPACE.sub('+', '+'.join(filter(None.__ne__, map(
                lambda x:
                    None
                    if (pd.isnull(x[0]) or len(x[0])==0)
                    else str(x[0]) + x[1],
                zip(r[:-1], ('/', '', ',', ',', ''))
            )))) + '+%.0f' % r[-1]
        )

    def request(url, r):
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
                    match = self.street_number_processing(row['house_number'])
                    print(
                        'No matches: %s|%s (%s) - %s' %
                        (match.group(1), match.group(2),
                         row['house_number'], others)
                    )
                return [np.NaN, np.NaN, False]

            hit = matching_hits[0]['_source']
            return [float(hit['Y']), float(hit['X']), self.check_hit(hit, row)]

    ST_NO_REGEX = re.compile(r'.*/+\s*([^/]*)\s*|\s*([^/]*)\s*')
    def street_number_processing(self, x):
        return self.ST_NO_REGEX.match(str(x).lower())

    def filter_matches(self, a, b):
        match = self.street_number_processing(b)
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

    def progress_summary(self, start_time, current_row, file_length):
        if current_row != 0:
            elapsed_time = time.time() - start_time
            frac_complete = current_row/file_length
            est_time = min(elapsed_time * (1/frac_complete - 1), 8640000)
            print(
                '%.3f, %i addresses indexed, elapsed time %s, est. time remaining %s'
                % (
                    frac_complete,
                    current_row,
                    time.strftime('%H:%M:%S', time.gmtime(elapsed_time)),
                    time.strftime('%H:%M:%S', time.gmtime(est_time))
                )
            )
