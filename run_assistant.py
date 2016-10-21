import os
import datetime
from real_estate.url_manager import RealEstateUrlManager
from real_estate.settings import Settings
from real_estate.website_scraper import WebsiteScraper


class RunAssistant(object):
    def __init__(self, scraper,
                 base_url, min_page_num, max_page_num,
                 settings_file_path, run_type, run_dir):
        self.scraper = scraper
        self.url_manager = RealEstateUrlManager(
            base_url, min_page_num, max_page_num
        )

        self.settings = Settings(
            settings_file_path, run_type,
            run_dir, True
        )

    def run(self):
        WebsiteScraper.retrieve_and_json_all_pages(
            self.url_manager, self.settings.html_dump)
        html_mtime = datetime.datetime.fromtimestamp(
            os.path.getmtime(self.settings.html_dump))
        pages = WebsiteScraper.load_pages_from_json(
            self.settings.html_dump, self.settings.failures_log)
        properties = self.scraper.scrape_pages(pages)
        properties = WebsiteScraper.filter_scrapings(
            properties, self.settings.failures_log)
        df = WebsiteScraper.to_hdf(
            properties, self.settings.data_file, html_mtime)

        print('DataFrame summmary:\nShape %s\n%s\n' % (
            str(df.shape), str(df.dtypes)
        ))
        print('Scraping data from: %s' % str(html_mtime))
        print('Saved %i properties to: %s' %
              (len(properties), self.settings.data_file))
        print('Done.')
