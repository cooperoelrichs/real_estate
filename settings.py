from real_estate.json_load_and_dump import JSONLoadAndDump
import os


class Settings(object):
    def __init__(self, settings_file_path, run_type,
                 run_dir, verbosity):
        self.run_dir = run_dir
        self.verbosity = verbosity

        json_settings = JSONLoadAndDump.load_from_file(
            settings_file_path
        )
        run_settings = json_settings[run_type]
        geo_settings = json_settings['geo_settings']
        slack_settings = json_settings['slack_settings']

        data_dir = os.path.join(run_dir, run_settings['data_dir'])
        html_dir = os.path.join(run_dir, run_settings['html_dir'])
        self.make_dir_unless_exists(data_dir)
        self.make_dir_unless_exists(html_dir)

        self.data_file = os.path.join(data_dir, run_settings['data_file'])
        self.html_dump = os.path.join(html_dir, run_settings['html_dump'])
        self.failures_log = os.path.join(
            data_dir,
            run_settings['failures_log']
        )

        geo_data_dir = os.path.join(run_dir, geo_settings['geo_data_dir'])
        self.nsw_geo_equivs_file = os.path.join(
            geo_data_dir, geo_settings['nsw_geo_equivs_file']
        )

        self.general_channel = slack_settings['general_channel']
        self.exception_channel = slack_settings['exception_channel']
        self.slack_token = slack_settings['slack_token']
        
    def make_dir_unless_exists(self, dir_path):
        if not os.path.isdir(dir_path) and self.verbosity is True:
            print('Making the directory: %s' % dir_path)

        os.makedirs(dir_path, exist_ok=True)
