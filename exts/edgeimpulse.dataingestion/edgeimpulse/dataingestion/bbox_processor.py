import json
import numpy as np
from pathlib import Path

# creates info.labels file with bounding box data
def process_files(bounding_box_dir, rgb_dir, category):

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

    print(f"RGB images and info.labels saved in {rgb_dir}")
