import csv
import requests
from itertools import groupby
from functools import reduce
import pandas as pd
from scraper.page_scraper import PageScraper
from real_estate.json_load_and_dump import JSONLoadAndDump
from real_estate.data_processing.data_storer import DataStorer

from real_estate.memory_usage import MU


class WebsiteScraper():
    REQUEST_RETRIES = 20

    def to_hdf(properties, file_path, scrape_datetime):
        raise RuntimeError('removed.')

    def to_df(properties, scrape_datetime):
        if len(properties) == 0:
            raise RuntimeError('Properties list is empty.')

        data = [p.to_tuple() + (scrape_datetime,) for p in properties]
        column_names = properties[0].column_names() + ('date_scraped',)
        df = pd.DataFrame.from_records(data, columns=column_names)
        return df

    def update_data_store(df, file_type, file_path, scrape_time):
        DataStorer.create_new_unless_exists(df, file_type, file_path)
        DataStorer.update_data_store(df, file_type, file_path, scrape_time)

    def retrieve_and_json_pages_by_postcodes(url_manager, file_path, pcs):
        htmls = WebsiteScraper.retrieve_all_pages_for_postcodes(
            url_manager, pcs
        )

        WebsiteScraper.dump_htmls(
            {
                'htmls': htmls,
                'search_description': (
                    'base URL: %s, max page number: %i\npostcodes: %s'
                    % (url_manager.base_url, url_manager.maximum_page_number,
                       ', '.join([str(pc) for pc in pcs]))
                )
            },
            file_path
        )

    def retrieve_and_json_all_pages(url_manager, file_path):
        MU.print_memory_usage('04.01')
        htmls = WebsiteScraper.retrieve_all_pages(url_manager)
        MU.print_memory_usage('04.02')
        WebsiteScraper.dump_htmls(
            {
                'htmls': htmls,
                'search_description': (
                    'base URL: %s, max page number: %i'
                    % (url_manager.base_url, url_manager.maximum_page_number)
                )
            },
            file_path
        )
        MU.print_memory_usage('04.03')

    def dump_htmls(htmls, file_path):
        JSONLoadAndDump.dump_to_file(htmls, file_path)

    def load_pages_from_json(file_path):
        htmls = JSONLoadAndDump.load_from_file(file_path)
        return htmls

    def filter_scrapings(scrapings, log_file_path):
        valids, invalids = WebsiteScraper.split_scrapings(scrapings)
        WebsiteScraper.log_failures(invalids, log_file_path)
        return valids, invalids

    def retrieve_all_pages_for_postcodes(url_manager, pcs):
        htmls = []
        for i, (pc, state) in enumerate(pcs):
            print(
                'Retrieving pages for postcode %i in %s, number %i of %i' %
                (pc, state, i+1, len(pcs))
            )
            htmls += WebsiteScraper.retrieve_htmls_for_postcode(
                    url_manager, pc, state
            )
        return htmls

    def retrieve_htmls_for_postcode(url_manager, pc, state):
        htmls = []
        for i in range(url_manager.maximum_page_number):
            page_num = i + 1
            url = url_manager.make_url_for_page_and_postcode(page_num, pc, state)
            html = WebsiteScraper.retrieve_html_page(url)

            if PageScraper.no_results_check(
                PageScraper.html_to_soup(html), page_num
            ):
                return htmls
            else:
                htmls.append(html)
        return htmls

    def retrieve_all_pages(url_manager, verbose=False):
        htmls = []
        for i in range(url_manager.maximum_page_number):
            page_num = i + 1
            if verbose:
                print('Retrieving page %i.' % page_num)
            url = url_manager.make_url_for_page(page_num)
            html = WebsiteScraper.retrieve_html_page(url)

            if PageScraper.no_results_check(
                PageScraper.html_to_soup(html), page_num
            ):
                return htmls
            else:
                htmls.append(html)
        return htmls

    def retrieve_html_page(url):
        response = WebsiteScraper.attempt_to_retrieve_page(
            url, 0, WebsiteScraper.REQUEST_RETRIES
        )
        html = response.text
        return html

    def attempt_to_retrieve_page(url, attempts, max_attempts):
        error = None
        for i in range(max_attempts):
            try:
                response = requests.get(url, timeout=1)
                response.raise_for_status()
                return response
            except requests.exceptions.Timeout as e:
                error = e
                pass
            except requests.exceptions.ReadTimeout as e:
                error = e
                pass
            except requests.exceptions.ConnectionError as e:
                error = e
                pass
            except requests.exceptions.ConnectTimeout as e:
                error = e
                pass
            except requests.exceptions.HTTPError as e:
                error = e
                print(error)
            print('Trying to get page again, attempt %i.' % i)
        raise error

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
