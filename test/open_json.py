import json


def open_json_file(file_path):
    x = None
    with open(file_path, 'r') as f:
        x = json.load(f)
    return x
