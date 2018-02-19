import re
import requests
import numpy as np


class ElasticsearchServer(object):
    def __init__(self):
        self.start_server()
        pass

    def start_server():
        pass

    def stop_server():
        pass


class StreetscopeServer(object):
    def __init__(self):
        self.start_server()
        pass

    def start_server():
        pass

    def stop_server():
        pass


class Geocoder(object):
    def __init__(self):
        pass


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
        coords = data[self.INDICIES].apply(
            self.geocode, raw=True, axis=1
        )

    def geocode(self, r):
        url = self.mk_url(r)
        result = StreetscopeGeocoder.request(url)
        self.process_result(result)

        # check result
        # extract coords
        # return coords

    def mk_url(self, r):
        return (
            'http://localhost:5000/geocode?query=' +
            self.RE_SPACE.sub('+', '+'.join(filter(None.__ne__, map(
                lambda x: None if x[0] is None else str(x[0]) + x[1],
                zip(r, ('/', '', ',', ',', '', ''))
            ))))
        )

    def request(url):
        current_delay = 0.1
        max_delay = 2

        while True:
            try:
                response = requests.get(url)
            except IOError:
                pass
            else:
                return response.json()

            if current_delay < max_delay:
                raise Exception('Too many retry attempts.')
            time.sleep(current_delay)
            current_delay *= 2

    def process_result(self, r):
        if len(r['features']) == 0:
            print('No results.')
            pass
        else:
            feature = r['features'][0]
            geo = feature['geometry']
            if geo['type'] != 'Point':
                raise ValueError('Not a supported geo type: %s' % geo['type'])

            lat, lon = geo['coordinates']
            address_str = feature['properties']['formatted_address']
            print(address_str)


    # Notes:
    # - do we rely on the user to start the streetscope app and elasticsearch
    #   or do we start them and close them here;
    # - start this code afresh because it will need to be fast and reliable;
    # - this will require some thorough tests of the results returned by
    #   streetscope to ensure we have the correct location.
