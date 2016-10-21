from real_estate.json_load_and_dump import JSONLoadAndDump
import os


class Settings(object):
    def __init__(self, settings_file_path, run_type,
                 url_manager, run_dir, verbosity):
        self.url_manager = url_manager
        self.run_dir = run_dir
        self.verbosity = verbosity

        json_settings = JSONLoadAndDump.load_from_file(
            settings_file_path)[run_type]

        data_dir = os.path.join(run_dir, json_settings['data_dir'])
        html_dir = os.path.join(run_dir, json_settings['html_dir'])
        self.make_dir_unless_exists(data_dir)
        self.make_dir_unless_exists(html_dir)

        self.data_file = os.path.join(data_dir, json_settings['data_file'])
        self.html_dump = os.path.join(html_dir, json_settings['html_dump'])
        self.failures_log = os.path.join(
            data_dir,
            json_settings['failures_log']
        )

    def make_dir_unless_exists(self, dir_path):
        if not os.path.isdir(dir_path) and self.verbosity is True:
            print('Making the directory: %s' % dir_path)

        print('MAKING %s' % dir_path)
        os.makedirs(dir_path, exist_ok=True)
