import json
import os


class ConfigManager:
    def __init__(self, filename="config.json"):
        self.filename = filename

    def save_params(self, params):
        data = self._load()
        data["params"] = params
        self._save(data)

    def load_params(self):
        data = self._load()
        return data["params"]

    def save_threshold(self, threshold):
        data = self._load()
        data["threshold"] = threshold
        self._save(data)

    def load_threshold(self):
        data = self._load()
        return data["params"]

    def _save(self, data):
        with open(self.filename, "w") as f:
            json.dump(data, f, indent=4)

    def _load(self):
        if not os.path.exists(self.filename):
            with open(self.filename, "w") as f:
                f.write("{}")

        with open(self.filename, "r") as f:
            return json.load(f)
