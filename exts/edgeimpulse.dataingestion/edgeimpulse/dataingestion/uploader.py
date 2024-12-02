# uploader.py
import asyncio
import requests
import os

async def upload_data(
    api_key,
    data_folder,
    dataset,
    log_callback,
    on_sample_upload_success,
    on_upload_complete,
):
    dataset_types = ["training", "testing", "anomaly"]
    if dataset not in dataset_types:
        log_callback(
            f"Error: Dataset type invalid (must be training, testing, or anomaly). Provided: {dataset}"
        )
        return

    url = "https://ingestion.edgeimpulse.com/api/" + dataset + "/files"
    bounding_boxes_file_path = os.path.join(data_folder, "bounding_boxes.labels")

    try:
        for file in os.listdir(data_folder):
            file_path = os.path.join(data_folder, file)
            label = os.path.basename(file_path).split(".")[0]

            if os.path.isfile(file_path):
                await asyncio.sleep(1) 

                try:
                    with open(file_path, "rb") as file_data:
                        files = [
                            ("data", (os.path.basename(file_path), file_data, "image/png"))
                        ]

                        # if bounding_boxes.labels exists, open and append it to the files list
                        if os.path.isfile(bounding_boxes_file_path):
                            bbox_file = open(bounding_boxes_file_path, "rb") 
                            files.append(
                                ("data", ("bounding_boxes.labels", bbox_file, "multipart/form-data"))
                            )

                        try:
                            res = requests.post(
                                url=url,
                                headers={
                                    "x-label": label,
                                    "x-api-key": api_key,
                                    "x-disallow-duplicates": "1",
                                },
                                files=files,
                            )

                            if res.status_code == 200:
                                log_callback(f"Success: {file_path} uploaded successfully.")
                                on_sample_upload_success()
                            else:
                                log_callback(
                                    f"Error: {file_path} failed to upload. Status Code {res.status_code}: {res.text}"
                                )
                        finally:
                            if 'bbox_file' in locals():
                                bbox_file.close()

                except Exception as e:
                    log_callback(
                        f"Error: Failed to process {file_path}. Exception: {str(e)}"
                    )
    except FileNotFoundError:
        log_callback("Error: Data Path invalid.")

    log_callback("Done")
    on_upload_complete()