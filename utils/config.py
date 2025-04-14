# config.py

import yaml
import os


class Config:
    _config_data = None

    @classmethod
    def load_config(cls, file_path="./utils/config.yaml"):
        if cls._config_data is None:
            with open(file_path, "r") as f:
                cls._config_data = yaml.safe_load(f)
        return cls._config_data

    # Access method for constants
    def get_config():
        return Config.load_config()
