import csv
import requests
from itertools import groupby
from functools import reduce
import bs4
import pandas as pd
from scraper.page_scraper import PageScraper
from real_estate.json_load_and_dump import JSONLoadAndDump
from real_estate.data_processing.data_storer import DataStorer

from real_estate.memory_usage import MU


class WebsiteScraper():
    def to_hdf(properties, file_path, scrape_datetime):
        raise RuntimeError('removed.')

    def to_df(properties, scrape_datetime):
        if len(properties) == 0:
            raise RuntimeError('Properties list is empty.')

        data = [p.to_tuple() + (scrape_datetime,) for p in properties]
        column_names = properties[0].column_names() + ('date_scraped',)
        df = pd.DataFrame.from_records(data, columns=column_names)
        return df

    def parse_addresses_separately(properties):
        return PAP.parse(properties)

    def update_data_store(df, file_path):
        DataStorer.create_new_unless_exists(df, file_path)
        DataStorer.update_data_store(df, file_path)

    def retrieve_and_json_pages_by_postcodes(url_manager, file_path, pcs):
        soups = WebsiteScraper.retrieve_all_pages_for_postcodes(url_manager, pcs)
        WebsiteScraper.dump_soups(soups, file_path)

    def retrieve_and_json_all_pages(url_manager, file_path):
        soups = WebsiteScraper.retrieve_all_pages(url_manager)
        WebsiteScraper.dump_soups(soups, file_path)

    def dump_soups(soups, file_path):
        htmls = [str(soup) for soup in soups]
        JSONLoadAndDump.dump_to_file(htmls, file_path)

    def load_pages_from_json(file_path):
        MU.print_memory_usage()
        htmls = JSONLoadAndDump.load_from_file(file_path)
        MU.print_memory_usage()
        pages = [bs4.BeautifulSoup(html, "html.parser") for html in htmls]
        MU.print_memory_usage()
        return pages

    def filter_scrapings(scrapings, log_file_path):
        valids, invalids = WebsiteScraper.split_scrapings(scrapings)
        # WebsiteScraper.report_on_failures(invalids)
        WebsiteScraper.log_failures(invalids, log_file_path)
        return valids, invalids

    def retrieve_all_pages_for_postcodes(url_manager, pcs):
        soups = []
        for i, (pc, state) in enumerate(pcs):
            print(
                'Retrieving pages for postcode %i in %s, number %i of %i' %
                (pc, state, i+1, len(pcs))
            )
            soups += WebsiteScraper.retrieve_soups_for_postcode(
                url_manager, pc, state
            )
        return soups

    def retrieve_soups_for_postcode(url_manager, pc, state):
        soups = []
        for i in range(url_manager.maximum_page_number):
            page_num = i + 1
            url = url_manager.make_url_for_page_and_postcode(page_num, pc, state)
            soup = WebsiteScraper.retrieve_soup_for_a_single_page(url)

            if PageScraper.no_results_check(soup, page_num):
                return soups
            else:
                soups.append(soup)

    def retrieve_all_pages(url_manager):
        soups = []
        for i in range(url_manager.maximum_page_number):
            page_num = i + 1
            print('Retrieving page %i.' % page_num)
            url = url_manager.make_url_for_page(page_num)
            soup = WebsiteScraper.retrieve_soup_for_a_single_page(url)

            if PageScraper.no_results_check(soup, page_num):
                return soups
            else:
                soups.append(soup)

        return soups

    def retrieve_html_page(url):
        response = WebsiteScraper.attempt_to_retrieve_page(url, 0, 20)
        response.raise_for_status()
        html = response.text
        return html

    def attempt_to_retrieve_page(url, attempts, max_attempts):
        for _ in range(max_attempts):
            try:
                return requests.get(url, timeout=1)
            except requests.exceptions.Timeout as e:
                pass
            except requests.exceptions.ConnectionError as e:
                pass
            print('trying to get page again.')
        raise e


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
