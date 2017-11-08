from real_estate.json_load_and_dump import JSONLoadAndDump
import os


class Settings(object):
    def __init__(self, settings_file_path):
        self.json = self.load_settings_file(settings_file_path)

    def load_settings_file(self, fp):
        return JSONLoadAndDump.load_from_file(fp)

    def make_dir_unless_exists(self, dir_path):
        if not os.path.isdir(dir_path) and self.verbosity is True:
            print('Making the directory: %s' % dir_path)
        os.makedirs(dir_path, exist_ok=True)


class AssistantSettings(Settings):
    def __init__(self, state, run_category, settings_file_path,
                 run_dir, verbose):
        super().__init__(settings_file_path)
        self.run_dir = run_dir
        self.verbose = verbose
        state_settings = self.json[state]
        run_category_settings = state_settings[run_category]

        data_dir = os.path.join(self.run_dir, self.json['data_dir'])
        html_dir = os.path.join(self.run_dir, self.json['html_dir'])
        self.make_dir_unless_exists(data_dir)
        self.make_dir_unless_exists(html_dir)

        self.data_file_type = run_category_settings['data_file_type']
        self.data_file = os.path.join(data_dir, run_category_settings['data_file'])
        self.html_dump = os.path.join(html_dir, run_category_settings['html_dump'])
        self.failures_log = os.path.join(data_dir, run_category_settings['failures_log'])

        self.by_postcode = state_settings['by_postcode']
        if self.by_postcode:
            self.postcodes_sample_size = state_settings['postcodes_sample_size']
            self.postcodes_file_path = os.path.join(
                run_dir, self.json['geo_data_dir'], state_settings['postcodes_file']
            )

        self.base_url = run_category_settings['url']
        self.max_page_number = state_settings['max_page_number']


class SlackSettings(Settings):
    def __init__(self, settings_file_path):
        super().__init__(settings_file_path)
        slack_settings = self.json['slack_settings']
        self.general_channel = slack_settings['general_channel']
        self.exception_channel = slack_settings['exception_channel']
        self.slack_token = slack_settings['slack_token']
