class RealEstateUrlManager(object):
    def __init__(self, base, rent_or_buy,
                 search, page_post_fix,
                 first_page, maximum_page_number):
        self.base = base
        self.rent_or_buy = rent_or_buy
        self.search = search
        self.page_post_fix = page_post_fix
        self.first_page = first_page
        self.maximum_page_number = maximum_page_number

    def make_url_for_page(self, number):
        url = '%s/%s/%s/%s-%i' % (
            self.base, self.rent_or_buy,
            self.search, self.page_post_fix, number
        )
        return url
