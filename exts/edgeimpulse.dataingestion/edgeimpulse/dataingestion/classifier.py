import json
import subprocess
from enum import Enum, auto
import os
import tempfile
import shutil
import numpy as np
import glob
from omni.kit.widget.viewport.capture import ByteCapture
import omni.isaac.core.utils.viewports as vp
import ctypes
from PIL import Image

from .client import EdgeImpulseRestClient


class ClassifierError(Enum):
    SUCCESS = auto()
    NODEJS_NOT_INSTALLED = auto()
    FAILED_TO_RETRIEVE_PROJECT_ID = auto()
    MODEL_DEPLOYMENT_NOT_AVAILABLE = auto()
    FAILED_TO_DOWNLOAD_MODEL = auto()
    FAILED_TO_PROCESS_VIEWPORT = auto()


class Classifier:
    def __init__(self, projectApiKey, log_fn):
        self.restClient = EdgeImpulseRestClient(projectApiKey)
        self.projectId = None
        self.modelReady = False
        # TODO allow users to specify output dir for models
        self.modelPath = os.path.expanduser("~/Desktop/model.zip")
        self.featuresTmpFile = None
        self.log_fn = log_fn

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

    async def check_and_update_model(self):
        if not self.is_node_installed():
            self.log_fn("Error: NodeJS not installed")
            return ClassifierError.NODEJS_NOT_INSTALLED

        self.projectId = await self.restClient.get_project_id()
        if self.projectId is None:
            self.log_fn("Error: Failed to retrieve project ID")
            return ClassifierError.FAILED_TO_RETRIEVE_PROJECT_ID

        deployment_info = await self.restClient.get_deployment_info(self.projectId)
        if not deployment_info:
            self.log_fn("Error: Failed to get deployment info")
            return ClassifierError.MODEL_DEPLOYMENT_NOT_AVAILABLE

        current_version = deployment_info["version"]
        model_dir_name = f"ei-model-{current_version}"
        model_dir = os.path.join(os.path.dirname(self.modelPath), model_dir_name)

        # Check if the model directory exists and its version
        if os.path.exists(model_dir):
            self.log_fn("Latest model version already downloaded.")
            self.modelReady = True
            return ClassifierError.SUCCESS

        # If the model directory for the current version does not exist, delete old versions and download the new one
        self.delete_old_models(os.path.dirname(self.modelPath), model_dir_name)
        self.log_fn("Downloading model...")
        model_content = await self.restClient.download_model(self.projectId)
        if model_content is not None:
            self.save_model(model_content, model_dir)
            self.modelReady = True
            self.log_fn("Model is ready for classification.")
            return ClassifierError.SUCCESS
        else:
            self.log_fn("Error: Failed to download the model")
            return ClassifierError.FAILED_TO_DOWNLOAD_MODEL

    def delete_old_models(self, parent_dir, exclude_dir_name):
        for dirname in os.listdir(parent_dir):
            if dirname.startswith("ei-model-") and dirname != exclude_dir_name:
                dirpath = os.path.join(parent_dir, dirname)
                shutil.rmtree(dirpath)
                self.log_fn(f"Deleted old model directory: {dirpath}")

    def save_model(self, model_content, model_dir):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            self.log_fn(f"'{model_dir}' directory created")

        model_zip_path = os.path.join(model_dir, "model.zip")
        with open(model_zip_path, "wb") as model_file:
            model_file.write(model_content)
        self.log_fn(f"Model zip saved to {model_zip_path}")

        # Extract the zip file
        shutil.unpack_archive(model_zip_path, model_dir)
        os.remove(model_zip_path)
        self.log_fn(f"Model extracted to {model_dir}")

    def resize_image_and_extract_features(
        self, image, target_width, target_height, channel_count
    ):
        # Resize the image
        img_resized = image.resize(
            (target_width, target_height), Image.Resampling.LANCZOS
        )

        # Convert the image to the required color space
        if channel_count == 3:  # RGB
            img_resized = img_resized.convert("RGB")
            img_array = np.array(img_resized)
            # Extract RGB features as hexadecimal values
            features = [
                "0x{:02x}{:02x}{:02x}".format(*pixel)
                for pixel in img_array.reshape(-1, 3)
            ]
        elif channel_count == 1:  # Grayscale
            img_resized = img_resized.convert("L")
            img_array = np.array(img_resized)
            # Repeat the grayscale values to mimic the RGB structure
            features = [
                "0x{:02x}{:02x}{:02x}".format(pixel, pixel, pixel)
                for pixel in img_array.flatten()
            ]

        return {
            "features": features,
            "originalWidth": image.width,
            "originalHeight": image.height,
            "newWidth": target_width,
            "newHeight": target_height,
        }

    async def capture_and_process_image(self):
        def on_capture_completed(buffer, buffer_size, width, height, format):
            try:
                image_size = width * height * 4
                ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.POINTER(
                    ctypes.c_byte * image_size
                )
                ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [
                    ctypes.py_object,
                    ctypes.c_char_p,
                ]
                content = ctypes.pythonapi.PyCapsule_GetPointer(buffer, None)
                pointer = ctypes.cast(
                    content, ctypes.POINTER(ctypes.c_byte * image_size)
                )
                np_arr = np.ctypeslib.as_array(pointer.contents)
                image = Image.frombytes("RGBA", (width, height), np_arr.tobytes())

                # Directly use the image for resizing and feature extraction
                # TODO test values, get them from impulse
                target_width = 320
                target_height = 320
                channel_count = 3  # 3 for RGB, 1 for grayscale
                resized_info = self.resize_image_and_extract_features(
                    image, target_width, target_height, channel_count
                )
                features = resized_info["features"]
                features_str = ",".join(features)

                try:
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".txt", mode="w+t"
                    ) as tmp_file:
                        tmp_file.write(features_str)
                        self.featuresTmpFile = tmp_file.name
                        self.log_fn(f"Features saved to {tmp_file.name}")
                except Exception as e:
                    self.log_fn(f"Error: Failed to save features to file: {e}")

            except Exception as e:
                self.log_fn(f"Error: Failed to process and save image: {e}")

        viewport_window_id = vp.get_id_from_index(0)
        viewport_window = vp.get_window_from_id(viewport_window_id)
        viewport_api = viewport_window.viewport_api

        capture_delegate = ByteCapture(on_capture_completed)
        capture = viewport_api.schedule_capture(capture_delegate)

        await capture.wait_for_result()

    async def classify(self):
        self.log_fn("Checking and updating model...")
        result = await self.check_and_update_model()
        if result != ClassifierError.SUCCESS:
            self.log_fn(f"Failed to update model: {result.name}")
            return result

        self.log_fn("Capturing and processing image...")
        await self.capture_and_process_image()

        try:
            model_dirs = glob.glob(
                os.path.join(os.path.dirname(self.modelPath), "ei-model-*")
            )
            if model_dirs:
                latest_model_dir = max(model_dirs, key=os.path.getctime)
                self.log_fn(f"Using latest model directory: {latest_model_dir}")

                if self.featuresTmpFile:
                    script_dir = os.path.join(latest_model_dir, "node")
                    self.log_fn(f"Running inference on {self.featuresTmpFile}")
                    process = subprocess.run(
                        ["node", "run-impulse.js", self.featuresTmpFile],
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        cwd=script_dir,
                    )
                    self.log_fn(process.stdout)
                    return ClassifierError.SUCCESS
                else:
                    return ClassifierError.FAILED_TO_PROCESS_VIEWPORT
            else:
                self.log_fn("No model directory found.")
                return ClassifierError.FAILED_TO_DOWNLOAD_MODEL
        except subprocess.CalledProcessError as e:
            self.log_fn(f"Error executing model classification: {e.stderr}")
            return ClassifierError.FAILED_TO_DOWNLOAD_MODEL
