# SPDX-License-Identifier: Apache-2.0

import omni.ext
import omni.ui as ui
from omni.kit.window.file_importer import get_file_importer
import asyncio
import requests
import os

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

        self._window = ui.Window("Edge Impulse Data Ingestion", width=450, height=220)
        with self._window.frame:
            with ui.VStack(spacing=8):
                
                with ui.HStack(height=20):
                    ui.Spacer(width=3)
                    ui.Label("Create a free Edge Impulse account: https://studio.edgeimpulse.com/", height=20, word_wrap=True)
                    ui.Spacer(width=3)

                with ui.HStack(height=20):
                    ui.Spacer(width=3)
                    ui.Label("Data Path", width=70)
                    ui.Spacer(width=8)
                    self.data_path_display = ui.Label("No folder selected", width=250)
                    ui.Button("Select Folder", clicked_fn=self.select_folder)
                    ui.Spacer(width=3)

                with ui.HStack(height=20):
                    ui.Spacer(width=3)
                    ei_api_key_label = ui.Label("API Key", width=70)
                    ui.Spacer(width=8)
                    ei_api_key = ui.StringField(name="ei_api_key")
                    ei_api_key.model.set_value("ei_02162...")
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

                def on_click():
                    #asyncio.ensure_future()
                    loop = asyncio.get_event_loop()
                    res = loop.run_until_complete(upload_data(self.API_KEY, self.DATA_FOLDER, self.DATASET))
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
                self.DATA_FOLDER = dirname
            else:
                print("No folder selected")

        file_importer = get_file_importer()
        file_importer.show_window(
            title="Select Data Folder",
            show_only_folders=True,
            import_handler=import_handler,
            import_button_label="Select"
        )


    def on_shutdown(self):
        print("[edgeimpulse.dataingestion] Edge Impulse Data Ingestion shutdown")

   

    
