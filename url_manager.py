class RealEstateUrlManager(object):
    def __init__(self, base_url, first_page, maximum_page_number):
        self.base_url = base_url
        self.first_page = first_page
        self.maximum_page_number = maximum_page_number

    def make_url_for_page(self, number):
        return '%s-%i' % (self.base_url, number)
