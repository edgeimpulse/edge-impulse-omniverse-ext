import os
import json

from .state import State


class Config:
    def __init__(self, config_file="config.json"):
        self.config_file = config_file
        self.config_data = self.load_config()

    def load_config(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, "r") as file:
                return json.load(file)
        return {}

    def print_config_info(self):
        """Prints the path and content of the configuration file."""
        print(f"Config file path: {self.config_file}")
        print("Config contents:")
        print(json.dumps(self.config_data, indent=4))

    def save_config(self):
        with open(self.config_file, "w") as file:
            json.dump(self.config_data, file)

    def get(self, key, default=None):
        return self.config_data.get(key, default)

    def set(self, key, value):
        self.config_data[key] = value
        self.save_config()

    def get_state(self):
        """Retrieve the saved state from the config."""
        # Default to NO_PROJECT_CONNECTED if not set
        return self.get("state", State.NO_PROJECT_CONNECTED.name)

    def set_state(self, state):
        """Save the current state to the config."""
        self.set("state", state.name)
