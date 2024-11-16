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
from .bbox_processor import process_files

class EdgeImpulseExtension(omni.ext.IExt):

    config = Config()
    classifier = None

    def on_startup(self, ext_id):
        print("[edgeimpulse.dataingestion] Edge Impulse Extension startup")

        self.config.print_config_info()

        # Load the last known state from the config
        saved_state_name = self.config.get_state()
        try:
            self.state = State[saved_state_name]
        except KeyError:
            self.state = State.NO_PROJECT_CONNECTED

        self.reset_to_initial_state()

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

    def reset_to_initial_state(self):
        self.project_id = None
        self.api_key = None
        self.project_name = None

        self.impulse = None

        self.upload_logs_text = ""
        self.uploading = False

        self.classify_logs_text = ""
        self.classifying = False

        self.training_samples = 0
        self.testing_samples = 0
        self.anomaly_samples = 0

        self.impulse_info = None
        self.deployment_info = None

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
            self.setup_data_upload_ui()

            # Classification Section
            self.setup_classification_ui()

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
        self.reset_to_initial_state()
        self.config.set("project_id", None)
        self.config.set("project_name", None)
        self.config.set("project_api_key", None)
        self.classify_button.visible = False
        self.ready_for_classification.visible = False
        self.data_collapsable_frame.collapsed = True
        self.classification_collapsable_frame.collapsed = True
        self.transition_to_state(State.NO_PROJECT_CONNECTED)

    ### Data ingestion

    def setup_data_upload_ui(self):
        self.data_collapsable_frame = ui.CollapsableFrame(
            "Data Upload", collapsed=True, height=0
        )
        self.data_collapsable_frame.set_collapsed_changed_fn(
            lambda c: asyncio.ensure_future(self.on_data_upload_collapsed_changed(c))
        )
        with self.data_collapsable_frame:
            with ui.VStack(spacing=10, height=0):
                with ui.VStack(height=0, spacing=5):
                    self.training_samples_label = ui.Label(
                        f"Training samples: {self.training_samples}"
                    )
                    self.testing_samples_label = ui.Label(
                        f"Test samples: {self.testing_samples}"
                    )
                    self.anomaly_samples_label = ui.Label(
                        f"Anomaly samples: {self.anomaly_samples}"
                    )
                ui.Spacer(height=10)

                with ui.HStack(height=10):
                    ui.Spacer(width=3)
                    ui.Label("Add Bounding Boxes", width=70)
                    ui.Spacer(width=5)
                    self.checkbox = ui.CheckBox(width=20, height=20)
                    self.checkbox.model.set_value(False)
                    self.checkbox.model.add_value_changed_fn(self.on_checkbox_changed)
                    ui.Spacer(width=3)

                with ui.HStack(height=20):
                    ui.Spacer(width=3)

                    self.path_label = ui.Label("Data Path", width=70) # switch between "Data Path" and "RGB Path"

                    ui.Spacer(width=8)
                    data_path = self.config.get("data_path", "No folder selected")
                    self.data_path_display = ui.Label(data_path, width=250)
                    ui.Spacer(width=10)
                    ui.Button("Select Folder", clicked_fn=lambda: self.select_folder("data"), width=150)
                    ui.Spacer(width=3)

                self.bounding_box_path = ui.HStack(visible=False, height=20)
                with self.bounding_box_path:
                    ui.Spacer(width=3)
                    ui.Label("Bounding Box Path", width=70)
                    ui.Spacer(width=8)
                    bbox_data_path = self.config.get("bbox_data_path", "No folder selected")
                    self.bbox_path_display = ui.Label(bbox_data_path, width=250)
                    ui.Spacer(width=10)
                    ui.Button("Select Folder", clicked_fn=lambda: self.select_folder("bbox"), width=150)
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

    def on_checkbox_changed(self, model):
        self.bounding_box_path.visible = model.as_bool

        if model.as_bool:
            self.path_label.text = "RGB Path"
        else:
            self.path_label.text = "Data Path"

    async def on_data_upload_collapsed_changed(self, collapsed):
        if not collapsed:
            await self.get_samples_count()

    def select_folder(self, path_type="data"):
        def import_handler(filename: str, dirname: str, selections: list = []):
            if dirname:
                if path_type == "data":
                    self.data_path_display.text = dirname
                    EdgeImpulseExtension.config.set("data_path", dirname)
                elif path_type == "bbox":
                    self.bbox_path_display.text = dirname
                    EdgeImpulseExtension.config.set("bbox_data_path", dirname)
            else:
                print("No folder selected")

        file_importer = get_file_importer()
        file_importer.show_window(
            title="Select Folder",
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
        selected_index = self.dataset_type_dropdown.model.get_item_value_model().as_int
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

        # create info.labels file
        process_files(self.config.get("bbox_data_path"), self.config.get("data_path"), self.get_dataset_type())

        if not self.uploading:  # Prevent multiple uploads at the same time
            self.uploading = True
            self.upload_button.text = "Uploading..."
            self.upload_logs_frame.visible = True

            async def upload():
                await upload_data(
                    self.config.get("project_api_key"),
                    self.config.get("data_path"),
                    self.config.get("dataset_type"),
                    self.add_upload_logs_entry,
                    lambda: asyncio.ensure_future(self.get_samples_count()),
                    self.on_upload_complete,
                )

            asyncio.ensure_future(upload())

    def on_upload_complete(self):
        self.uploading = False
        self.upload_button.text = "Upload to Edge Impulse"

    async def get_samples_count(self):
        self.training_samples = await self.rest_client.get_samples_count(
            self.project_id, "training"
        )
        self.testing_samples = await self.rest_client.get_samples_count(
            self.project_id, "testing"
        )
        self.anomaly_samples = await self.rest_client.get_samples_count(
            self.project_id, "anomaly"
        )
        print(
            f"Samples count: Training ({self.training_samples}) - Testing ({self.testing_samples}) - Anomaly ({self.anomaly_samples})"
        )
        self.training_samples_label.text = f"Training samples: {self.training_samples}"
        self.testing_samples_label.text = f"Test samples: {self.testing_samples}"
        self.anomaly_samples_label.text = f"Anomaly samples: {self.anomaly_samples}"

    ### Classification

    def setup_classification_ui(self):
        self.classification_collapsable_frame = ui.CollapsableFrame(
            "Classification", collapsed=True, height=0
        )
        self.classification_collapsable_frame.set_collapsed_changed_fn(
            lambda c: asyncio.ensure_future(self.on_classification_collapsed_changed(c))
        )
        with self.classification_collapsable_frame:
            with ui.VStack(spacing=10, height=0):
                self.impulse_status_label = ui.Label(
                    "Fetching your Impulse design...", height=20, visible=False
                )

                self.deployment_status_label = ui.Label(
                    "Fetching latest model deployment...", height=20, visible=False
                )

                self.ready_for_classification = ui.Label(
                    "Your model is ready! You can now run inference on the current scene",
                    height=20,
                    visible=False,
                )

                ui.Spacer(height=20)

                with ui.HStack(height=20):
                    self.classify_button = ui.Button(
                        "Classify current scene frame",
                        clicked_fn=lambda: asyncio.ensure_future(self.start_classify()),
                        visible=False,
                    )

                # Scrolling frame for classify logs
                self.classify_logs_frame = ui.ScrollingFrame(height=100, visible=False)
                with self.classify_logs_frame:
                    self.classify_logs_label = ui.Label("", word_wrap=True)

                with ui.HStack(height=20):
                    self.clear_classify_logs_button = ui.Button(
                        "Clear Logs", clicked_fn=self.clear_classify_logs, visible=False
                    )

                self.classification_output_section = ui.CollapsableFrame(
                    "Ouput", collapsed=True, visible=False, height=0
                )
                with self.classification_output_section:
                    self.image_display = ui.Image(
                        "",
                        width=400,
                        height=300,
                    )
                    self.image_display.visible = False

    async def on_classification_collapsed_changed(self, collapsed):
        if not collapsed:
            if not self.impulse_info:
                self.impulse_status_label.visible = True
                self.impulse_status_label.text = "Fetching your Impulse design..."
                self.impulse_info = await self.rest_client.get_impulse(self.project_id)
                if not self.impulse_info:
                    self.impulse_status_label.text = f"""Your Impulse is not ready yet.\n
Go to https://studio.edgeimpulse.com/studio/{self.project_id}/create-impulse to configure and train your model"""
                    return
                if self.impulse_info.input_type != "image":
                    self.impulse_info = None
                    self.impulse_status_label.text = "Invalid Impulse input block type. Only 'image' type is supported"
                    return

            self.impulse_status_label.text = "Impulse is ready"
            self.impulse_status_label.visible = False

            if not self.deployment_info or not self.deployment_info.has_deployment:
                self.deployment_status_label.visible = True
                self.deployment_status_label.text = (
                    "Fetching your latest model deployment..."
                )
                self.deployment_info = await self.rest_client.get_deployment_info(
                    self.project_id
                )
                if not self.deployment_info.has_deployment:
                    self.deployment_status_label.text = f"""Your model WebAssembly deployment is not ready yet.\n
Go to https://studio.edgeimpulse.com/studio/{self.project_id}/deployment to build a WebAssembly deployment"""
                    return
            self.deployment_status_label.text = "Model deployment ready"
            self.deployment_status_label.visible = False

            if self.impulse_info and self.deployment_info:
                self.classify_button.visible = True
                self.ready_for_classification.visible = True
            else:
                self.classify_button.visible = False
                self.ready_for_classification.visible = False

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

    async def get_impulse(self):
        self.impulse = await self.rest_client.get_impulse(self.project_id)

    async def start_classify(self):
        if not self.classifier:
            if not self.impulse:
                await self.get_impulse()

            if not self.impulse:
                self.add_classify_logs_entry("Error: impulse is not ready yet")
                return

            self.classifier = Classifier(
                self.rest_client,
                self.project_id,
                self.impulse.image_height,
                self.impulse.image_width,
                self.add_classify_logs_entry,
            )

        async def classify():
            try:
                self.classifying = True
                self.classify_button.text = "Classifying..."
                self.clear_classify_logs()
                self.classification_output_section.visible = False
                image_path = await self.classifier.classify()
                corrected_path = image_path[1].replace("\\", "/")
                self.image_display.source_url = corrected_path
                self.image_display.width = ui.Length(self.impulse_info.image_width)
                self.image_display.height = ui.Length(self.impulse_info.image_height)
                self.image_display.visible = True
                self.classification_output_section.visible = True
                self.classification_output_section.collapsed = False
            finally:
                self.classifying = False
                self.classify_button.text = "Classify"

        asyncio.ensure_future(classify())

    def on_shutdown(self):
        print("[edgeimpulse.dataingestion] Edge Impulse Extension shutdown")
