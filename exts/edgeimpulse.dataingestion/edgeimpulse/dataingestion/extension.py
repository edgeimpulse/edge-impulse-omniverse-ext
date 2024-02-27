# SPDX-License-Identifier: Apache-2.0

import omni.ext
import omni.ui as ui
from omni.kit.window.file_importer import get_file_importer
import asyncio

from .config import Config
from .uploader import upload_data
from .classifier import Classifier
from .state import State
from .client import EdgeImpulseRestClient


class EdgeImpulseExtension(omni.ext.IExt):

    config = Config()
    classifier = None
    restClient = None

    def on_startup(self, ext_id):
        print("[edgeimpulse.dataingestion] Edge Impulse Extension startup")

        self.config.print_config_info()

        # Load the last known state from the config
        saved_state_name = self.config.get_state()
        try:
            self.state = State[saved_state_name]
        except KeyError:
            self.state = State.NO_PROJECT_CONNECTED

        self.project_id = None
        self.api_key = None
        self.project_name = None

        self.upload_logs_text = ""
        self.uploading = False

        self.classify_logs_text = ""
        self.classifying = False

        self._window = ui.Window("Edge Impulse", width=300, height=300)

        with self._window.frame:
            with ui.VStack():
                self.no_project_content_area = ui.VStack(
                    spacing=15, height=0, visible=False
                )
                self.project_connected_content_area = ui.VStack(
                    spacing=15, height=0, visible=False
                )

        self.setup_ui_project_connected()
        self.setup_ui_no_project_connected()

        self.transition_to_state(self.state)

    def transition_to_state(self, new_state):
        """
        Transition the extension to a new state.

        :param new_state: The new state to transition to.
        """
        # Store the new state in memory
        self.state = new_state

        # Save the new state in the configuration
        self.config.set_state(self.state)

        # Call the corresponding UI setup method based on the new state
        if self.state == State.PROJECT_CONNECTED:
            self.project_id = self.config.get("project_id")
            self.project_name = self.config.get("project_name")
            self.api_key = self.config.get("project_api_key")
            self.rest_client = EdgeImpulseRestClient(self.api_key)
            self.project_info_label.text = (
                f"Connected to project {self.project_id} ({self.project_name})"
            )

        self.update_ui_visibility()

    def update_ui_visibility(self):
        """Update UI visibility based on the current state."""
        if hasattr(self, "no_project_content_area") and hasattr(
            self, "project_connected_content_area"
        ):
            self.no_project_content_area.visible = (
                self.state == State.NO_PROJECT_CONNECTED
            )
            self.project_connected_content_area.visible = (
                self.state == State.PROJECT_CONNECTED
            )

    def setup_ui_no_project_connected(self):
        with self.no_project_content_area:
            # Title and welcome message
            ui.Label(
                "Welcome to Edge Impulse for NVIDIA Omniverse",
                height=20,
                word_wrap=True,
            )

            ui.Label(
                "1. Create a free Edge Impulse account: https://studio.edgeimpulse.com/",
                height=20,
                word_wrap=True,
            )

            # API Key input section
            with ui.VStack(height=20, spacing=10):
                ui.Label(
                    "2. Connect to your Edge Impulse project by setting your API Key",
                    width=300,
                )
                with ui.HStack():
                    ui.Spacer(width=3)
                    ei_api_key = ui.StringField(name="ei_api_key", height=20)
                    ui.Spacer(width=3)
                ui.Spacer(width=30)

            with ui.HStack(height=20):
                ui.Spacer(width=30)
                # Connect button
                connect_button = ui.Button("Connect")
                connect_button.set_clicked_fn(
                    lambda: asyncio.ensure_future(
                        self.validate_and_connect_project(
                            ei_api_key.model.get_value_as_string()
                        )
                    )
                )
                ui.Spacer(width=30)

            self.error_message_label = ui.Label(
                "", height=20, word_wrap=True, visible=False
            )

    def setup_ui_project_connected(self):
        with self.project_connected_content_area:
            # Project information
            self.project_info_label = ui.Label(
                f"Connected to project {self.project_id} ({self.project_name})",
                height=20,
                word_wrap=True,
            )

            # Disconnect button
            with ui.HStack(height=20):
                ui.Spacer(width=30)
                disconnect_button = ui.Button("Disconnect")
                disconnect_button.set_clicked_fn(lambda: self.disconnect())
                ui.Spacer(width=30)

            # Data Upload Section
            with ui.CollapsableFrame("Data Upload", collapsed=True, height=0):
                with ui.VStack(spacing=10, height=0):
                    self.setup_data_upload_ui()

            # Classification Section
            with ui.CollapsableFrame("Classification", collapsed=True, height=0):
                with ui.VStack(spacing=10, height=0):
                    self.setup_classification_ui()

    def setup_data_upload_ui(self):
        with ui.HStack(height=20):
            ui.Spacer(width=3)
            ui.Label("Data Path", width=70)
            ui.Spacer(width=8)
            data_path = self.config.get("data_path", "No folder selected")
            self.data_path_display = ui.Label(data_path, width=250)
            ui.Spacer(width=10)
            ui.Button("Select Folder", clicked_fn=self.select_folder, width=150)
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

        # Scrolling frame for upload logs
        self.upload_logs_frame = ui.ScrollingFrame(height=100, visible=False)
        with self.upload_logs_frame:
            self.upload_logs_label = ui.Label("", word_wrap=True)

        with ui.HStack(height=20):
            self.clear_upload_logs_button = ui.Button(
                "Clear Logs", clicked_fn=self.clear_upload_logs, visible=False
            )

    def setup_classification_ui(self):
        with ui.HStack(height=20):
            self.classify_button = ui.Button(
                "Classify",
                clicked_fn=lambda: asyncio.ensure_future(self.start_classify()),
            )

        # Scrolling frame for classify logs
        self.classify_logs_frame = ui.ScrollingFrame(height=100, visible=False)
        with self.classify_logs_frame:
            self.classify_logs_label = ui.Label("", word_wrap=True)

        with ui.HStack(height=20):
            self.clear_classify_logs_button = ui.Button(
                "Clear Logs", clicked_fn=self.clear_classify_logs, visible=False
            )

        self.image_display = ui.Image(
            "",
            width=400,
            height=300,
        )
        self.image_display.visible = False

    def hide_error_message(self):
        if self.error_message_label:
            self.error_message_label.text = ""
            self.error_message_label.visible = False

    def display_error_message(self, message):
        if self.error_message_label:
            self.error_message_label.text = message
            self.error_message_label.visible = True

    async def validate_and_connect_project(self, api_key):
        self.hide_error_message()

        self.rest_client = EdgeImpulseRestClient(api_key)
        project_info = await self.rest_client.get_project_info()

        if project_info:
            print(f"Connected to project: {project_info}")
            self.config.set("project_id", project_info["id"])
            self.config.set("project_name", project_info["name"])
            self.config.set("project_api_key", api_key)
            self.transition_to_state(State.PROJECT_CONNECTED)
        else:
            # Display an error message in the current UI
            self.display_error_message(
                "Failed to connect to the project. Please check your API key."
            )

    def disconnect(self):
        print("Disconnecting")
        self.project_id = None
        self.project_name = None
        self.api_key = None
        self.config.set("project_id", None)
        self.config.set("project_name", None)
        self.config.set("project_api_key", None)
        self.transition_to_state(State.NO_PROJECT_CONNECTED)

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

    def add_upload_logs_entry(self, message):
        self.upload_logs_text += message + "\n"
        self.upload_logs_label.text = self.upload_logs_text
        self.update_clear_upload_logs_button_visibility()

    def clear_upload_logs(self):
        self.upload_logs_text = ""
        self.upload_logs_label.text = self.upload_logs_text
        self.update_clear_upload_logs_button_visibility()

    def update_clear_upload_logs_button_visibility(self):
        self.clear_upload_logs_button.visible = bool(self.upload_logs_text)
        self.upload_logs_frame.visible = self.uploading

    def start_upload(self):
        if not self.uploading:  # Prevent multiple uploads at the same time
            self.uploading = True
            self.upload_button.text = "Uploading..."
            self.upload_logs_frame.visible = True

            async def upload():
                await upload_data(
                    self.config.get("api_key"),
                    self.config.get("data_path"),
                    self.config.get("dataset_type"),
                    self.add_upload_logs_entry,
                    self.on_upload_complete,
                )

            asyncio.ensure_future(upload())

    def on_upload_complete(self):
        self.uploading = False
        self.upload_button.text = "Upload to Edge Impulse"

    def add_classify_logs_entry(self, message):
        self.classify_logs_text += message + "\n"
        self.classify_logs_label.text = self.classify_logs_text
        self.update_clear_classify_logs_button_visibility()

    def clear_classify_logs(self):
        self.classify_logs_text = ""
        self.classify_logs_label.text = self.classify_logs_text
        self.update_clear_classify_logs_button_visibility()

    def update_clear_classify_logs_button_visibility(self):
        self.clear_classify_logs_button.visible = bool(self.classify_logs_text)
        self.classify_logs_frame.visible = self.classifying

    async def start_classify(self):
        if not self.classifier:
            self.classifier = Classifier(
                self.rest_client, self.project_id, self.add_classify_logs_entry
            )

        async def classify():
            try:
                self.classifying = True
                self.classify_button.text = "Classifying..."

                image_path = await self.classifier.classify()
                corrected_path = image_path[1].replace("\\", "/")
                self.image_display.source_url = corrected_path
                self.image_display.visible = True
            finally:
                self.classifying = False
                self.classify_button.text = "Classify"

        asyncio.ensure_future(classify())

    def on_shutdown(self):
        print("[edgeimpulse.dataingestion] Edge Impulse Extension shutdown")
