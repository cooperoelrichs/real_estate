class RealEstateUrlManager(object):
    POSTFIX = '?includeSurrounding=false&source=location-search'

    def __init__(self, base_url, first_page, maximum_page_number):
        self.base_url = base_url
        self.first_page = first_page
        self.maximum_page_number = maximum_page_number

    def make_url_for_page(self, number):
        # http://www.realestate.com.au/buy/in-act/list-1
        return '%s-%i' % (self.base_url, number)

    def make_url_for_page_and_postcode(self, number, pc, state):
        # http://www.realestate.com.au/buy/
        #     in-2000+nsw/
        #     list-1
        #     ?includeSurrounding=false&source=location-search

        return '{base_url:s}in-{pc:d}+{state:s}/list-{number:d}{postfix:s}'.format(
            base_url=self.base_url,
            pc=pc,
            state=state,
            number=number,
            postfix=self.POSTFIX
        )
