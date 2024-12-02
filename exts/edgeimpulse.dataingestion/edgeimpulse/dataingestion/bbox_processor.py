import json
import numpy as np
from pathlib import Path

# creates bounding_boxes.labels file with bounding box data
def process_files(bounding_box_dir, rgb_dir, log_callback):
    log_callback(f"Creating bounding_boxes.labels file...")

    bounding_box_dir = Path(bounding_box_dir)
    rgb_dir = Path(rgb_dir)

    # extract bbox type from bbox path (either tight or loose)
    bounding_box_dir_name = bounding_box_dir.name
    if bounding_box_dir_name.startswith("bounding_box_2d_"):
        label_type = bounding_box_dir_name.replace("bounding_box_2d_", "")
    else:
        raise ValueError(f"Invalid bounding box directory name: {bounding_box_dir_name}")

    # data structure for bounding_boxes.labels
    bounding_boxes_labels_data = {"version": 1, "type": "bounding-box-labels", "boundingBoxes": {}}

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

        # load unique rgb file path
        rgb_image_file = f"rgb_{file_number}.png"

        # prepare bbox data for bounding_boxes.labels
        bounding_boxes_entry = []
        for i, bbox in enumerate(bounding_boxes):
            label = labels[str(i)]["class"]
            x_min = int(bbox["x_min"])
            y_min = int(bbox["y_min"])
            width = int(bbox["x_max"] - x_min)
            height = int(bbox["y_max"] - y_min)

            # add bbox to entry structure
            bounding_box_dict = {"label": label, "x": x_min, "y": y_min, "width": width, "height": height}
            bounding_boxes_entry.append(bounding_box_dict)

        # add entry to bounding_boxes.labels structure
        bounding_boxes_labels_data["boundingBoxes"][rgb_image_file] = bounding_boxes_entry

    # write bounding_boxes.labels file to the same directory as the rgb files
    bounding_boxes_labels_path = rgb_dir / "bounding_boxes.labels"
    with open(bounding_boxes_labels_path, "w") as f:
        json.dump(bounding_boxes_labels_data, f, indent=4)

    log_callback(f"Success: bounding_boxes.labels file in {rgb_dir}")

# deletes bounding_boxes.labels that was created by the above program
def post_process_files(rgb_dir, log_callback):
    log_callback(f"Deleting bounding_boxes.labels file from {rgb_dir}...")

    rgb_dir = Path(rgb_dir)
    bounding_boxes_labels_path = rgb_dir / "bounding_boxes.labels"

    if bounding_boxes_labels_path.exists():
        bounding_boxes_labels_path.unlink()
        log_callback(f"Success: Deleted bounding_boxes.labels file from {rgb_dir}")
    else:
        log_callback(f"No bounding_boxes.labels file found in {rgb_dir}")