class Geocoder(object):
    """
    Geocode addresses using Streetscope.
    streetscope: https://github.com/codeforamerica/streetscope
    """

    # Notes:
    # - do we rely on the user to start the streetscope app and elasticsearch
    #   or do we start them and close them here;
    # - start this code afresh because it will need to be fast and reliable;
    # - this will require some thorough tests of the results returned by
    #   streetscope to ensure we have the correct location.

    # Plan:
    # 1. build address url;
    # 2. fetch result;
    # 3. parse result (manually or using libpostal?);
    # 4. tripple check we got the correct address;
    # 5. add the lat and lon to the df.
