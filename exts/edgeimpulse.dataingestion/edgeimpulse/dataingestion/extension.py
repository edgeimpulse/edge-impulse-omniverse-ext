# SPDX-License-Identifier: Apache-2.0

import omni.ext
import omni.ui as ui
import omni.kit
import asyncio
import aiohttp
from omni.ui import style_utils
from functools import partial
import requests
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from omni.kit.viewport.utility import get_active_viewport_window

async def upload_data(api_key, data_folder, dataset):
    dataset_types = ["training", "testing", "anomaly"]
    if dataset not in dataset_types:
        return('Error: Dataset type invalid (must be training, testing, or anomaly).')
    url = 'https://ingestion.edgeimpulse.com/api/' + dataset + '/files'
    try:
        for file in os.listdir(data_folder):
            file_path = os.path.join(data_folder, file)
            # Labels are determined from the filename, anything after "." is ignored, i.e.
            # File "object.1.blah.png" will be uploaded as file object.1.blah with label "object"
            label = os.path.basename(file_path).split('.')[0]
            if os.path.isfile(file_path):
                with open(file_path, 'r') as file:
                    res = requests.post(url=url,
                    headers= {
                        'x-label': label,
                        'x-api-key': api_key,
                        'x-disallow-duplicates': '1'
                    },
                    files = { 'data': (os.path.basename(file_path), open(file_path, 'rb'), 'image/png') }
                )
            if (res.status_code == 200):
                return(res.text)
            else:
                return(str("Error: Status Code " + str(res.status_code) + ": " + res.text))
    except FileNotFoundError:
        return('Error: Data Path invalid.')
    
async def get_trained_model(api_key, data_folder):
    return(str("get_trained_model clicked"))

async def inference_current_frame(api_key, data_folder):
    # Load the TensorFlow Lite model
    model_path = os.path.join(data_folder, "object_detection.lite")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(input_details)

    # Capture viewport
    res = "inference_current_frame clicked"
    capture_extension = omni.kit.capture.viewport.CaptureExtension.get_instance()
    capture_extension.options._output_folder = data_folder
    capture_extension.options._file_type = '.png'
    capture_extension.options._res_width = input_details[0]['shape'][1]
    capture_extension.options._res_height = input_details[0]['shape'][2]
    try:
        capture_extension.start()
    except Exception as e:
        res = str("Error: " + f"{e}")    

    try:
        # Load and preprocess the input image
        image_path = os.path.join(data_folder, "Capture1.png")
        input_image = Image.open(image_path)
        input_image = input_image.convert('RGB')
        #input_image = tf.image.decode_png(tf.io.read_file(image_path))
        #input_image = input_image.resize(input_details[0]['shape'][1], input_details[0]['shape'][2])
        input_image = np.array(input_image)
        input_image = input_image.astype(np.float32)
        input_image = np.expand_dims(input_image, axis=0)
        #input_image = tf.reshape(input_image, [1, input_details[0]['shape'][1], input_details[0]['shape'][2], 4])
        print("Input image shape:", input_image.shape)

        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], input_image)

        # Perform inference
        interpreter.invoke()

        rects = interpreter.get_tensor(output_details[0]['index'])
        scores = interpreter.get_tensor(output_details[2]['index'])
        
        # Get the output tensor
        #output_tensor = interpreter.get_tensor(output_details[0]['index'])

        # Post-process the output
        res = str("Rects: " + str(rects) + "\nScores: " + str(scores))
    except Exception as e:
        res = str("Error: " + f"{e}")
    return(str(res))

def getPath(label, text):
    DataIngestion.DATA_FOLDER = os.path.normpath(text)
    #print("Data Path Changed:", DataIngestion.DATA_FOLDER)

def getEIAPIKey(label, text):
    DataIngestion.API_KEY = text
    #print("Edge Impulse API Key Changed:", DataIngestion.API_KEY)

def getDatasetType(label, text):
    DataIngestion.DATASET = text
    #print("Edge Impulse API Key Changed:", DataIngestion.DATASET)

# Any class derived from `omni.ext.IExt` in top level module (defined in `python.modules` of `extension.toml`) will be
# instantiated when extension gets enabled and `on_startup(ext_id)` will be called. Later when extension gets disabled
# on_shutdown() is called.
class DataIngestion(omni.ext.IExt):
    # ext_id is current extension id. It can be used with extension manager to query additional information, like where
    # this extension is located on filesystem.
    
    API_KEY = ''
    DATA_FOLDER = ''
    DATASET = 'training'

    def on_startup(self, ext_id):
        print("[edgeimpulse.dataingestion] Edge Impulse Data Ingestion startup")

        self._window = ui.Window("Edge Impulse", width=450, height=220)
        with self._window.frame:
            with ui.VStack(spacing=8):
                
                with ui.HStack(height=20):
                    ui.Spacer(width=3)
                    ui.Label("Create a free Edge Impulse account: https://studio.edgeimpulse.com/", height=20, word_wrap=True)
                    ui.Spacer(width=3)

                with ui.HStack(height=20):
                    ui.Spacer(width=3)
                    data_path_label = ui.Label("Data Path", width=70)
                    ui.Spacer(width=8)
                    data_path = ui.StringField(name="path")
                    data_path.model.set_value("C:\\Temp") ## TODO fix before push
                    data_path.model.add_value_changed_fn(lambda m, label=data_path_label: getPath(data_path_label, m.get_value_as_string()))
                    ui.Spacer(width=3)

                with ui.HStack(height=20):
                    ui.Spacer(width=3)
                    ei_api_key_label = ui.Label("API Key", width=70)
                    ui.Spacer(width=8)
                    ei_api_key = ui.StringField(name="ei_api_key")
                    ei_api_key.model.set_value("ei_4713f2b0666f431cc1f31c410f3a8ad2f1f6684fc5666e90e22db308fba62e56") ## TODO fix before push
                    ei_api_key.model.add_value_changed_fn(lambda m, label=ei_api_key_label: getEIAPIKey(ei_api_key_label, m.get_value_as_string()))
                    ui.Spacer(width=3)

                with ui.HStack(height=20):
                    ui.Spacer(width=3)
                    dataset_type_label = ui.Label("Dataset", width=70)
                    ui.Spacer(width=8)
                    dataset_type = ui.StringField(name="dataset")
                    dataset_type.model.set_value("training, testing, or anomaly (default = training)")
                    dataset_type.model.add_value_changed_fn(lambda m, label=dataset_type_label: getDatasetType(dataset_type_label, m.get_value_as_string()))
                    ui.Spacer(width=3)

                def on_upload_data():
                    #asyncio.ensure_future()
                    loop = asyncio.get_event_loop()
                    res = loop.run_until_complete(upload_data(self.API_KEY, self.DATA_FOLDER, self.DATASET))
                    results_label.text = res

                def on_get_trained_model():
                    #asyncio.ensure_future()
                    loop = asyncio.get_event_loop()
                    res = loop.run_until_complete(get_trained_model(self.API_KEY, self.DATA_FOLDER))
                    results_label.text = res

                def on_inference_current_frame():
                    #asyncio.ensure_future()
                    loop = asyncio.get_event_loop()
                    res = loop.run_until_complete(inference_current_frame(self.API_KEY, self.DATA_FOLDER))
                    results_label.text = res
                
                with ui.HStack(height=20):
                    ui.Button("Upload to Edge Impulse", clicked_fn=on_upload_data)

                with ui.HStack(height=20):
                    ui.Spacer(width=3)
                    ui.Label("Train your model on edgeimpulse.com, then:", height=20, word_wrap=True)
                    ui.Spacer(width=3)

                with ui.HStack(height=20):
                    ui.Button("Get trained model", clicked_fn=on_get_trained_model)
                
                with ui.HStack(height=20):
                    ui.Button("Inference current frame", clicked_fn=on_inference_current_frame)

                with ui.HStack(height=20):
                    ui.Spacer(width=3)
                    results_label = ui.Label("", height=20, word_wrap=True)
                

    def on_shutdown(self):
        print("[edgeimpulse.dataingestion] Edge Impulse Data Ingestion shutdown")

   

    