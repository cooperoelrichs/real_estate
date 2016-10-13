import json


class JSONLoadAndDump():
    def load_from_file(file_path):
        x = None
        with open(file_path, 'r') as f:
            x = json.load(f)
        return x

    def dump_to_file(data, file_path):
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
