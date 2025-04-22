import yaml
from typing import Any, Union


class ConfigEditor:
  def __init__(self, config_path: str):
    self.config_path = config_path
    self.config_data = self._load_config()

  def _load_config(self) -> dict:
    """Load YAML configuration from file."""
    with open(self.config_path, 'r', encoding='utf-8') as file:
      return yaml.safe_load(file)

  def save_config(self) -> None:
    """Write the current config data back to the file."""
    with open(self.config_path, 'w', encoding='utf-8') as file:
      yaml.safe_dump(self.config_data, file,
                     allow_unicode=True, sort_keys=False)

  def get(self, keys: Union[str, list[str]]) -> Any:
    """Retrieve a nested config value using dot notation or list of keys."""
    if isinstance(keys, str):
      keys = keys.split('.')

    data = self.config_data
    for key in keys:
      data = data.get(key)
      if data is None:
        return None
    return data

  def set(self, keys: Union[str, list[str]], value: Any) -> None:
    """Set a nested config value using dot notation or list of keys."""
    if isinstance(keys, str):
      keys = keys.split('.')

    data = self.config_data
    for key in keys[:-1]:
      if key not in data or not isinstance(data[key], dict):
        data[key] = {}
      data = data[key]
    data[keys[-1]] = value

  def delete(self, keys: Union[str, list[str]]) -> None:
    """Delete a key from the config using dot notation or list of keys."""
    if isinstance(keys, str):
      keys = keys.split('.')

    data = self.config_data
    for key in keys[:-1]:
      if key not in data:
        return
      data = data[key]
    data.pop(keys[-1], None)

  def reload(self):
    """Reload the config from disk."""
    self.config_data = self._load_config()

  def print(self):
    """Print current config for debug purposes."""
    print(yaml.safe_dump(self.config_data, allow_unicode=True, sort_keys=False))
