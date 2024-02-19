import subprocess
from enum import Enum, auto
import os
import zipfile
import tempfile
import numpy as np
from omni.kit.widget.viewport.capture import ByteCapture
import omni.isaac.core.utils.viewports as vp


from .client import EdgeImpulseRestClient


class ClassifierError(Enum):
    SUCCESS = auto()
    NODEJS_NOT_INSTALLED = auto()
    FAILED_TO_RETRIEVE_PROJECT_ID = auto()
    MODEL_DEPLOYMENT_NOT_AVAILABLE = auto()
    FAILED_TO_DOWNLOAD_MODEL = auto()


class Classifier:
    def __init__(self, projectApiKey):
        self.restClient = EdgeImpulseRestClient(projectApiKey)
        self.projectId = None
        self.modelReady = False
        self.modelPath = os.path.expanduser("~/Desktop/model.zip")

    def is_node_installed(self):
        try:
            subprocess.run(
                ["node", "--version"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    async def check_and_prepare_model(self):
        if not self.is_node_installed():
            print("NodeJS not installed")
            return ClassifierError.NODEJS_NOT_INSTALLED

        if self.projectId is None:
            print("Fetching project information...")
            self.projectId = await self.restClient.get_project_id()
            if self.projectId is None:
                print("Failed to retrieve project ID")
                return ClassifierError.FAILED_TO_RETRIEVE_PROJECT_ID

        model_dir = os.path.join(os.path.dirname(self.modelPath), "eimodel")
        if os.path.exists(model_dir) and os.listdir(model_dir):
            print("Model is already ready for classification.")
            self.modelReady = True
            return ClassifierError.SUCCESS

        print("Checking model availability...")
        if not await self.restClient.check_model_deployment(self.projectId):
            print("Model deployment not available")
            return ClassifierError.MODEL_DEPLOYMENT_NOT_AVAILABLE

        print("Downloading model ...")
        model_content = await self.restClient.download_model(self.projectId)
        if model_content is not None:
            print("Saving model...")
            self.save_model(model_content)
            self.modelReady = True
            print("Model is ready for classification.")
            return ClassifierError.SUCCESS
        else:
            print("Failed to download the model")
            return ClassifierError.FAILED_TO_DOWNLOAD_MODEL

    def save_model(self, model_content):
        try:
            eimodel_dir = os.path.join(os.path.dirname(self.modelPath), "eimodel")
            if not os.path.exists(eimodel_dir):
                os.makedirs(eimodel_dir)
                print(f"'eimodel' directory created at {eimodel_dir}")

            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(model_content)
                zip_path = tmp_file.name
            print(f"Model zip saved temporarily to {zip_path}")

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(eimodel_dir)
            print(f"Model extracted to {eimodel_dir}")

            os.remove(zip_path)
            print(f"Temporary model zip removed.")

        except Exception as e:
            print(f"Failed to save or extract model: {e}")

    async def capture_and_process_image(self):
        def on_capture_completed(buffer, buffer_size, width, height, format):
            print(f"Captured image resolution: {width} x {height}, Format: {format}")
            # TODO process image and run inference

        viewport_window_id = vp.get_id_from_index(0)
        viewport_window = vp.get_window_from_id(viewport_window_id)
        viewport_api = viewport_window.viewport_api

        capture_delegate = ByteCapture(on_capture_completed)
        capture = viewport_api.schedule_capture(capture_delegate)

        await capture.wait_for_result()

    async def classify(self):
        result = await self.check_and_prepare_model()
        if result != ClassifierError.SUCCESS:
            return result

        print("Capturing viewport...")
        await self.capture_and_process_image()

        try:
            script_dir = os.path.join(
                os.path.dirname(self.modelPath), "eimodel", "node"
            )

            process = subprocess.run(
                ["node", "run-impulse.js", "123"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=script_dir,
            )
            print(process.stdout)
            return ClassifierError.SUCCESS
        except subprocess.CalledProcessError as e:
            print(f"Error executing model classification: {e.stderr}")
            return ClassifierError.FAILED_TO_DOWNLOAD_MODEL
