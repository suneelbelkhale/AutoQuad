import yaml

def read_params(file_name):
    with open(file_name) as f:
        # use safe_load instead load
        dataMap = yaml.safe_load(f)
        return dataMap