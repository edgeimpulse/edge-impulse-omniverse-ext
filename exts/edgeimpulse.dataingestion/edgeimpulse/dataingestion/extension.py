# SPDX-License-Identifier: Apache-2.0

import omni.ext
import omni.ui as ui
from omni.kit.window.file_importer import get_file_importer
import asyncio
import requests
import os
import json


class Config:
    def __init__(self, config_file="config.json"):
        self.config_file = config_file
        self.config_data = self.load_config()

    def load_config(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, "r") as file:
                return json.load(file)
        return {}

    def save_config(self):
        with open(self.config_file, "w") as file:
            json.dump(self.config_data, file)

    def get(self, key, default=None):
        return self.config_data.get(key, default)

    def set(self, key, value):
        self.config_data[key] = value
        self.save_config()


async def upload_data(api_key, data_folder, dataset):
    dataset_types = ["training", "testing", "anomaly"]
    if dataset not in dataset_types:
        return "Error: Dataset type invalid (must be training, testing, or anomaly)."
    url = "https://ingestion.edgeimpulse.com/api/" + dataset + "/files"
    try:
        for file in os.listdir(data_folder):
            file_path = os.path.join(data_folder, file)
            # Labels are determined from the filename, anything after "." is ignored, i.e.
            # File "object.1.blah.png" will be uploaded as file object.1.blah with label "object"
            label = os.path.basename(file_path).split(".")[0]
            if os.path.isfile(file_path):
                with open(file_path, "r") as file:
                    res = requests.post(
                        url=url,
                        headers={
                            "x-label": label,
                            "x-api-key": api_key,
                            "x-disallow-duplicates": "1",
                        },
                        files={
                            "data": (
                                os.path.basename(file_path),
                                open(file_path, "rb"),
                                "image/png",
                            )
                        },
                    )
            if res.status_code == 200:
                return res.text
            else:
                return str(
                    "Error: Status Code " + str(res.status_code) + ": " + res.text
                )
    except FileNotFoundError:
        return "Error: Data Path invalid."


def getPath(label, text):
    EdgeImpulseExtension.config.set("data_path", text)
    label.text = text


def getEIAPIKey(label, text):
    EdgeImpulseExtension.config.set("api_key", text)


def getDatasetType(label, text):
    EdgeImpulseExtension.config.set("dataset_type", text)


class EdgeImpulseExtension(omni.ext.IExt):

    config = Config()

    def on_startup(self, ext_id):
        print("[edgeimpulse.dataingestion] Edge Impulse Data Ingestion startup")

        self._window = ui.Window("Edge Impulse Data Ingestion", width=450, height=220)
        with self._window.frame:
            with ui.VStack(spacing=8):

                with ui.HStack(height=20):
                    ui.Spacer(width=3)
                    ui.Label(
                        "Create a free Edge Impulse account: https://studio.edgeimpulse.com/",
                        height=20,
                        word_wrap=True,
                    )
                    ui.Spacer(width=3)

                with ui.HStack(height=20):
                    ui.Spacer(width=3)
                    ui.Label("Data Path", width=70)
                    ui.Spacer(width=8)
                    data_path = self.config.get("data_path", "No folder selected")
                    print("data_path", data_path)
                    self.data_path_display = ui.Label(data_path, width=250)
                    ui.Button("Select Folder", clicked_fn=self.select_folder, width=150)
                    ui.Spacer(width=3)

                with ui.HStack(height=20):
                    ui.Spacer(width=3)
                    ui.Label("API Key", width=70)
                    ui.Spacer(width=8)
                    api_key = self.config.get("api_key", "ei_02162...")
                    ei_api_key = ui.StringField(name="ei_api_key")
                    ei_api_key.model.set_value(api_key)
                    ei_api_key.model.add_value_changed_fn(
                        lambda m, label=ei_api_key: getEIAPIKey(
                            label, m.get_value_as_string()
                        )
                    )
                    ui.Spacer(width=3)

                with ui.HStack(height=20):
                    ui.Spacer(width=3)
                    ui.Label("Dataset", width=70)
                    ui.Spacer(width=8)
                    dataset_types = ["training", "testing", "anomaly"]
                    self.dataset_type_dropdown = ui.ComboBox(0, *dataset_types)
                    self.dataset_type_subscription = (
                        self.dataset_type_dropdown.model.subscribe_item_changed_fn(
                            self.on_dataset_type_changed
                        )
                    )
                    initial_dataset_type = self.config.get("dataset_type", "training")
                    if initial_dataset_type in dataset_types:
                        for i, dtype in enumerate(dataset_types):
                            if dtype == initial_dataset_type:
                                self.dataset_type_dropdown.model.get_item_value_model().as_int = (
                                    i
                                )
                                break
                    ui.Spacer(width=3)

                def on_click():
                    # asyncio.ensure_future()
                    loop = asyncio.get_event_loop()
                    res = loop.run_until_complete(
                        upload_data(self.API_KEY, self.DATA_FOLDER, self.DATASET)
                    )
                    results_label.text = res

                with ui.HStack(height=20):
                    ui.Button("Upload to Edge Impulse", clicked_fn=on_click)

                with ui.HStack(height=20):
                    ui.Spacer(width=3)
                    results_label = ui.Label("", height=20, word_wrap=True)

    def select_folder(self):
        def import_handler(filename: str, dirname: str, selections: list = []):
            if dirname:
                self.data_path_display.text = dirname
                EdgeImpulseExtension.config.set(
                    "data_path", dirname
                )  # Save the selected folder to the config
            else:
                print("No folder selected")

        file_importer = get_file_importer()
        file_importer.show_window(
            title="Select Data Folder",
            show_only_folders=True,
            import_handler=import_handler,
            import_button_label="Select",
        )

    def on_dataset_type_changed(
        self, item_model: ui.AbstractItemModel, item: ui.AbstractItem
    ):
        value_model = item_model.get_item_value_model(item)
        current_index = value_model.as_int
        dataset_type = ["training", "testing", "anomaly"][current_index]
        self.config.set("dataset_type", dataset_type)

    def get_dataset_type(self):
        selected_index = self.dataset_type_dropdown.model.get_value_as_int()
        dataset_types = ["training", "testing", "anomaly"]
        return dataset_types[selected_index]

    def on_shutdown(self):
        print("[edgeimpulse.dataingestion] Edge Impulse Data Ingestion shutdown")
