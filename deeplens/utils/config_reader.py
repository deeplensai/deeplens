import yaml

class ConfigReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.config = self._read_config()

    def _read_config(self):
        with open(self.file_path, 'r') as file:
            return yaml.safe_load(file)

    def get(self, key, default=None):
        return self.config.get(key, default)