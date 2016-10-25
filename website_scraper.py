import csv
import requests
from itertools import groupby
from functools import reduce
import bs4
import pandas as pd
from scraper.page_scraper import PageScraper
from real_estate.json_load_and_dump import JSONLoadAndDump


class WebsiteScraper(object):
    def to_hdf(properties, file_path, scrape_datetime):
        if len(properties) == 0:
            raise RuntimeError('Properties list is empty.')

        data = [p.to_tuple() + (scrape_datetime,) for p in properties]
        column_names = properties[0].column_names() + ('datetime',)
        df = pd.DataFrame.from_records(data, columns=column_names)
        df.to_hdf(file_path, 'properties', append=False)
        return df

    def retrieve_and_json_all_pages(url_manager, file_path):
        soups = WebsiteScraper.retrieve_all_pages(url_manager)
        htmls = [str(soup) for soup in soups]
        JSONLoadAndDump.dump_to_file(htmls, file_path)

    def load_pages_from_json(file_path, log_file_path):
        htmls = JSONLoadAndDump.load_from_file(file_path)
        pages = [bs4.BeautifulSoup(html, "html.parser") for html in htmls]
        return pages

    def filter_scrapings(scrapings, log_file_path):
        valids, invalids = WebsiteScraper.split_scrapings(scrapings)
        WebsiteScraper.report_on_failures(invalids)
        WebsiteScraper.log_failures(invalids, log_file_path)
        return valids, invalids

    def retrieve_all_pages(url_manager):
        soups = []
        for i in range(url_manager.maximum_page_number):
            page_num = i + 1
            print('Retrieving page %i.' % page_num)
            url = url_manager.make_url_for_page(page_num)
            soup = WebsiteScraper.retrieve_soup_for_a_single_page(url)

            no_results = PageScraper.check_for_no_results(soup)
            if no_results:
                print("Found the 'no results' page, final page is number %i" %
                      (page_num - 1))
                return soups
            else:
                soups.append(soup)

        return soups

    def retrieve_html_page(url):
        response = requests.get(url)
        response.raise_for_status()
        html = response.text
        return html

    def retrieve_soup_for_a_single_page(url):
        html = WebsiteScraper.retrieve_html_page(url)
        soup = bs4.BeautifulSoup(html, "html.parser")
        return soup

    def split_scrapings(scrapings):
        checked_scrapings = [(x.is_valid(), x) for x in scrapings]
        valids = [x for validity, x in checked_scrapings if validity]
        invalids = [x for validity, x in checked_scrapings if not validity]
        return valids, invalids

    def report_on_failures(scrapings):
        print('Reporting on %i failed parses:' % len(scrapings))
        named = WebsiteScraper.named_failures(scrapings)
        for i, (name, x) in enumerate(named):
            print('### %i - %s' % (i, name))
            print(x.summarise())
        print('---- Reporting finished.')

    def log_failures(scrapings, file_path):
        named = WebsiteScraper.named_failures(scrapings)
        with open(file_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow([
                'error_number', 'error_type', 'sale_type', 'address', 'details'
            ])

            for i, (name, x) in enumerate(named):
                writer.writerow(
                    [i, name] + [y.summarise() for y in x.ordered_attributes()]
                )

    def named_failures(scrapings):
        sorted_scrapings = sorted(scrapings, key=lambda x: x.error_type_name())
        return [(x.error_type_name(), x) for x in sorted_scrapings]
        # grouped = [[(k, x) for x in X] for k, X in groupby(
        #     sorted_scrapings, lambda x: x.error_type_name()
        # )]
        # flatened = reduce(lambda y1, y2: y1 + y2, grouped)
        # return flatened
