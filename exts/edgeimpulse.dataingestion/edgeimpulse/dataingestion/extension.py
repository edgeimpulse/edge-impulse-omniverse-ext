# SPDX-License-Identifier: Apache-2.0

import omni.ext
import omni.ui as ui
from omni.kit.window.file_importer import get_file_importer
import asyncio

from .config import Config
from .uploader import upload_data


class EdgeImpulseExtension(omni.ext.IExt):

    config = Config()

    def on_startup(self, ext_id):
        print("[edgeimpulse.dataingestion] Edge Impulse Data Ingestion startup")

        self.log_text = ""
        self.uploading = False

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
                        lambda m: self.config.set("api_key", m.get_value_as_string())
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

                with ui.HStack(height=20):
                    self.upload_button = ui.Button(
                        "Upload to Edge Impulse", clicked_fn=lambda: self.start_upload()
                    )

                # Scrolling Frame for Logs
                with ui.ScrollingFrame(height=100):
                    self.log_label = ui.Label("", word_wrap=True)

                with ui.HStack(height=20):
                    self.clear_button = ui.Button(
                        "Clear Logs", clicked_fn=self.clear_logs
                    )
                    self.clear_button.visible = False

    def select_folder(self):
        def import_handler(filename: str, dirname: str, selections: list = []):
            if dirname:
                self.data_path_display.text = dirname
                EdgeImpulseExtension.config.set("data_path", dirname)
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

    def add_log_entry(self, message):
        self.log_text += message + "\n"
        self.log_label.text = self.log_text
        self.update_clear_button_visibility()

    def clear_logs(self):
        self.log_text = ""
        self.log_label.text = self.log_text
        self.update_clear_button_visibility()

    def update_clear_button_visibility(self):
        self.clear_button.visible = bool(self.log_text)

    def start_upload(self):
        if not self.uploading:  # Prevent multiple uploads at the same time
            self.uploading = True
            self.upload_button.text = "Uploading..."

            async def upload():
                await upload_data(
                    self.config.get("api_key"),
                    self.config.get("data_path"),
                    self.config.get("dataset_type"),
                    self.add_log_entry,
                    self.on_upload_complete,
                )

            asyncio.ensure_future(upload())

    def on_upload_complete(self):
        self.uploading = False
        self.upload_button.text = "Upload to Edge Impulse"

    def on_shutdown(self):
        print("[edgeimpulse.dataingestion] Edge Impulse Data Ingestion shutdown")
