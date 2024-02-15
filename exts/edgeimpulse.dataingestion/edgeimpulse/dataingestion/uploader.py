# uploader.py
import asyncio
import requests
import os


async def upload_data(api_key, data_folder, dataset, log_callback, on_upload_complete):
    dataset_types = ["training", "testing", "anomaly"]
    if dataset not in dataset_types:
        log_callback(
            "Error: Dataset type invalid (must be training, testing, or anomaly)."
        )
        return

    url = "https://ingestion.edgeimpulse.com/api/" + dataset + "/files"
    try:
        for file in os.listdir(data_folder):
            file_path = os.path.join(data_folder, file)
            label = os.path.basename(file_path).split(".")[0]
            if os.path.isfile(file_path):
                await asyncio.sleep(1)  # Simulated delay for demonstration
                try:
                    with open(file_path, "rb") as file_data:
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
                                    file_data,
                                    "image/png",
                                )
                            },
                        )
                        if res.status_code == 200:
                            log_callback(f"Success: {file_path} uploaded successfully.")
                        else:
                            log_callback(
                                f"Error: {file_path} failed to upload. Status Code {res.status_code}: {res.text}"
                            )
                except Exception as e:
                    log_callback(
                        f"Error: Failed to process {file_path}. Exception: {str(e)}"
                    )
    except FileNotFoundError:
        log_callback("Error: Data Path invalid.")
    log_callback("Done")
    on_upload_complete()
