import json
import numpy as np
from pathlib import Path
import os
import edgeimpulse as ei

# creates info.labels file with bounding box data
def process_files(bounding_box_dir, rgb_dir, category, log_callback):

    log_callback(f"Creating info.labels file...")


    bounding_box_dir = Path(bounding_box_dir)
    rgb_dir = Path(rgb_dir)

    # extract bbox type from bbox path (either tight or loose)
    bounding_box_dir_name = bounding_box_dir.name
    if bounding_box_dir_name.startswith("bounding_box_2d_"):
        label_type = bounding_box_dir_name.replace("bounding_box_2d_", "")
    else:
        raise ValueError(f"Invalid bounding box directory name: {bounding_box_dir_name}")

    # data structure for info.labels
    info_labels_data = {"version": 1, "files": []}

    # load npy files
    npy_files = [f for f in bounding_box_dir.iterdir() if f.suffix == ".npy"]

    for npy_file in npy_files:
        file_number = npy_file.stem.split("_")[-1]

        # load bbox data
        bounding_boxes = np.load(npy_file, allow_pickle=True)

        # load labels from json
        json_label_file = f"bounding_box_2d_{label_type}_labels_{file_number}.json"
        json_label_path = bounding_box_dir / json_label_file
        with open(json_label_path, "r") as f:
            labels = json.load(f)

        # load rgb file path
        rgb_image_file = f"rgb_{file_number}.png"

        # prepare bbox data
        bounding_boxes_list = []
        for i, bbox in enumerate(bounding_boxes):
            label = labels[str(i)]["class"]
            x_min = int(bbox["x_min"])
            y_min = int(bbox["y_min"])
            width = int(bbox["x_max"] - x_min)
            height = int(bbox["y_max"] - y_min)

            # add bbox to data structure
            bounding_box_dict = {"label": label, "x": x_min, "y": y_min, "width": width, "height": height}
            bounding_boxes_list.append(bounding_box_dict)

        # add entry to info.labels structure
        file_data = {
            "path": rgb_image_file,
            "category": category,
            "label": {"type": "label", "label": "cubes"},
            "metadata": None,
            "boundingBoxes": bounding_boxes_list,
        }
        info_labels_data["files"].append(file_data)

    # write info.labels file to the same directory as the rgb files TODO: delete after upload
    info_labels_path = rgb_dir / "info.labels"
    with open(info_labels_path, "w") as f:
        json.dump(info_labels_data, f, indent=4)

    log_callback(f"Success: info.labels file in {rgb_dir}")

# deletes info.labels that was created by the above program
def post_process_files (rgb_dir, log_callback):
    log_callback(f"Deleting info.lables file from {rgb_dir}...")

    rgb_dir = Path(rgb_dir)
    info_labels_path = rgb_dir / "info.labels"

    if info_labels_path.exists():
        info_labels_path.unlink()
        log_callback(f"Success: Deleted info.lables file from {rgb_dir}")
    else:
        log_callback(f"No info.labels file found in {rgb_dir}")

def upload_with_sdk(labels_file_path, data_folder, log_callback, api_key):
    ei.API_KEY = api_key

    try:

        with open(labels_file_path, 'r') as f:
            labels_data = json.load(f)

         # ensure 'files' key exists
        if 'files' not in labels_data:
            print("Error: 'files' key not found in info.labels.")
            return

        # upload each file
        for entry in labels_data['files']:
            if 'path' not in entry:
                print("Error: Missing 'path' in one of the entries in info.labels.")
                continue

            file_path = os.path.join(data_folder, entry["path"])
            label = os.path.basename(file_path).split(".")[0]

            if os.path.isfile(file_path):

                with open(file_path, "rb") as file_data:

                    # sample object
                    sample = ei.experimental.data.Sample(
                        filename=os.path.basename(file_path),
                        data=file_data.read(),
                        label=label,
                        bounding_boxes=entry.get("boundingBoxes", []),
                        metadata=entry.get("metadata", {})
                    )

                    # upload using python SDK
                    response = ei.experimental.data.upload_samples([sample])

                    # check for failed uploads
                    if len(response.fails) == 0:
                        print(f"Success: {file_path} uploaded successfully.")
                    else:
                        print(f"Error: Could not upload {file_path}.")

    except FileNotFoundError:
        log_callback("Error: Data path invalid.")
    except json.JSONDecodeError:
        log_callback("Error: .labels file could not be parsed.")
    log_callback("Upload complete.")