class Config:
    def __init__(self, config_path="config.json"):
        import os
        import json
        self.config_path = config_path
        self.data = None
        self.read_config()

    def read_config(self):
        import json
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        except Exception as e:
            print(f"Error reading config: {e}")
            self.data = {}
        return self.data

    def write_config(self, config_path=None):
        path = config_path if config_path else self.config_path
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error writing config: {e}")

    def get(self, key, default=None):
        if self.data:
            return self.data.get(key, default)
        return default

    def set(self, key, value):
        if self.data is not None:
            self.data[key] = value
        else:
            self.data = {key: value}
