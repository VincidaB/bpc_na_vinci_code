import os
import argparse
import shutil
from tqdm import tqdm
import json
from PIL import Image


def create_symlinks(
    dataset_dir,
    split,
    scene_ids,
    modalities,
    output_dir,
):
    """
    creates symlinks for all images in dataset_dit/[scene_ids]/[modalities]/[].png or .tif
    Args:
        dataset_dir: str, path to the dataset directory
        split: str, name of the split (train_pbr, val, test)
        scene_ids: list of str, scene ids to create symlinks for
        modalities: list of str, modalities to create symlinks for
        output_dir: str, path to the output directory
    """

    for scene_id in scene_ids:
        for modality in modalities:
            scene_dir = os.path.join(dataset_dir, split, scene_id)
            modality_dir = os.path.join(scene_dir, modality)
            for img in os.listdir(modality_dir):
                img_path = os.path.join(modality_dir, img)
                # print(img_path)
                os.symlink(
                    img_path,
                    os.path.join(output_dir, scene_id + "_" + modality + "_" + img),
                )


def create_yaml_file(output_dir, classes_map):
    """
    creates a yaml file for the dataset
    Args:
        output_dir: str, path to the output directory
        classes_map: dict, mapping of class ids to continuous range
    """
    with open(os.path.join(output_dir, "ipd.yaml"), "w") as f:
        f.write("train: ../train/images\n")
        f.write("val: ../val/images\n")
        f.write("test: ../test/images\n")
        f.write("nc: " + str(len(classes_map)) + "\n")
        f.write("names: [")
        for i, class_id in enumerate(classes_map.keys()):
            f.write(f"'{class_id}'")
            if i < len(classes_map) - 1:
                f.write(", ")
        f.write("]\n")


def create_labels(dataset_dir, split, modalities, labels_dir, class_map):

    camera_gt_map = {
        "rgb_cam1": "scene_gt_cam1.json",
        "rgb_cam2": "scene_gt_cam2.json",
        "rgb_cam3": "scene_gt_cam3.json",
    }
    camera_gt_info_map = {
        "rgb_cam1": "scene_gt_info_cam1.json",
        "rgb_cam2": "scene_gt_info_cam2.json",
        "rgb_cam3": "scene_gt_info_cam3.json",
    }

    scene_folders = [
        d
        for d in os.listdir(os.path.join(dataset_dir, split))
        if os.path.isdir(os.path.join(dataset_dir, split, d)) and not d.startswith(".")
    ]
    description = "Processing " + split + " scenes"
    for scene_folder in tqdm(scene_folders, desc=description):
        scene_path = os.path.join(dataset_dir, split, scene_folder)

        # For each camera, read bounding box info
        for cam in modalities:
            rgb_path = os.path.join(scene_path, cam)
            scene_gt_file = os.path.join(scene_path, camera_gt_map[cam])
            scene_gt_info_file = os.path.join(scene_path, camera_gt_info_map[cam])

            if not os.path.exists(rgb_path):
                print(f"Missing RGB folder for {cam} in {scene_folder}: {rgb_path}")
                continue
            if not os.path.exists(scene_gt_file):
                print(f"Missing JSON file for {cam} in {scene_folder}: {scene_gt_file}")
                continue
            if not os.path.exists(scene_gt_info_file):
                print(
                    f"Missing JSON file for {cam} in {scene_folder}: {scene_gt_info_file}"
                )
                continue

            # Load the JSON files for ground truth + info
            with open(scene_gt_file, "r") as f:
                scene_gt_data = json.load(f)
            with open(scene_gt_info_file, "r") as f:
                scene_gt_info_data = json.load(f)

            # Assume image IDs go from 0..N-1
            num_imgs = len(scene_gt_data)  # or use max key from scene_gt_data
            for img_id in range(num_imgs):
                img_key = str(img_id)
                if split == "train_pbr":
                    img_file = os.path.join(rgb_path, f"{img_id:06d}.jpg")
                else:
                    img_file = os.path.join(rgb_path, f"{img_id:06d}.png")

                if not os.path.exists(img_file):
                    # If the image doesn't exist, skip
                    continue
                if img_key not in scene_gt_data or img_key not in scene_gt_info_data:
                    # If there's no ground-truth info for this frame, skip
                    continue

                # We want all the obj_ids
                # We also check if visibility fraction > 0 (you can adjust this threshold)
                valid_bboxes = []  # List to store (class_id, bbox) tuples

                for bbox_info, gt_info in zip(
                    scene_gt_info_data[img_key], scene_gt_data[img_key]
                ):
                    if bbox_info["visib_fract"] > 0:
                        class_id = gt_info["obj_id"]
                        valid_bboxes.append(
                            (class_id, bbox_info["bbox_obj"])
                        )  # (class_id, (x, y, w, h))

                if not valid_bboxes:
                    # No bounding boxes for our object in this image
                    print(
                        f"No valid bounding boxes for {scene_folder}_{cam}_{img_id:06d}"
                    )
                    continue

                # TODO : remove the other funcion and put the symlink or copy code here
                # Copy the image to the YOLO "images/" folder
                # out_img_name = f"{scene_folder}_{cam}_{img_id:06d}.jpg"
                # out_img_path = os.path.join(images_dir, out_img_name)
                # shutil.copy(img_file, out_img_path)

                # Read real image dimensions
                with Image.open(img_file) as img:
                    img_width, img_height = img.size

                # Write YOLO label(s) for all bounding boxes in this image
                out_label_name = f"{scene_folder}_{cam}_{img_id:06d}.txt"
                out_label_path = os.path.join(labels_dir, out_label_name)

                if len(class_map) != 10:
                    print(
                        f"Classes for {scene_folder}_{cam}_{img_id:06d} are not continuous: {class_map}"
                    )
                    continue

                with open(out_label_path, "w") as lf:
                    for class_id, (x, y, w, h) in valid_bboxes:
                        x_center = (x + w / 2) / img_width
                        y_center = (y + h / 2) / img_height
                        width = w / img_width
                        height = h / img_height
                        # YOLO format: class x_center y_center width height
                        # Here class is always '0' because we have only 1 object
                        lf.write(
                            f"{class_map[class_id]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                        )


def convert_ipd_to_yolo(output_dir):
    # Define the IPD dataset directory
    ipd_dir = "../../datasets/ipd"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir, ipd_dir)

    # Define the output directory
    output_dir = os.path.join(script_dir, "../../datasets/", output_dir)

    # check if the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        # check if the output directory is empty
        if len(os.listdir(output_dir)) != 0:
            user_input = input(
                f"Output directory already exists: {output_dir} and is not empty. Do you want to override its contents? (y/N): "
            )
            if user_input.lower() != "y":
                print("Exiting without making changes.")
                return
            else:
                # Clear the output directory
                for file in os.listdir(output_dir):
                    file_path = os.path.join(output_dir, file)
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        print(f"Removing directory {file_path}")
                        shutil.rmtree(file_path)

    # create a train, val and test directory
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")

    train_dir_images = os.path.join(train_dir, "images")
    train_dir_labels = os.path.join(train_dir, "labels")
    val_dir_images = os.path.join(val_dir, "images")
    val_dir_labels = os.path.join(val_dir, "labels")
    test_dir_images = os.path.join(test_dir, "images")
    test_dir_labels = os.path.join(test_dir, "labels")

    os.makedirs(train_dir)
    os.makedirs(val_dir)
    os.makedirs(test_dir)

    os.makedirs(train_dir_images)
    os.makedirs(val_dir_images)
    os.makedirs(test_dir_images)
    os.makedirs(train_dir_labels)
    os.makedirs(val_dir_labels)
    os.makedirs(test_dir_labels)

    train_pbr_scene_ids = [f"{i:06d}" for i in range(0, 10)]
    val_scene_ids = [f"{i:06d}" for i in range(0, 15)]
    test_scene_ids = [f"{i:06d}" for i in range(0, 15)]

    modalities = ["rgb_cam1", "rgb_cam2", "rgb_cam3"]

    # create symlinks of the images  in the train, val and test directories
    create_symlinks(
        dataset_dir, "train_pbr", train_pbr_scene_ids, modalities, train_dir_images
    )
    create_symlinks(dataset_dir, "val", val_scene_ids, modalities, val_dir_images)
    create_symlinks(dataset_dir, "test", test_scene_ids, modalities, test_dir_images)

    # now, onto turning the scene_gt_cam1.json files into yolo format labels

    class_map = {0: 0, 1: 1, 4: 2, 8: 3, 10: 4, 11: 5, 14: 6, 18: 7, 19: 8, 20: 9}

    create_labels(dataset_dir, "train_pbr", modalities, train_dir_labels, class_map)
    create_labels(dataset_dir, "val", modalities, val_dir_labels, class_map)
    create_labels(dataset_dir, "test", modalities, test_dir_labels, class_map)

    create_yaml_file(output_dir, class_map)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, required=True, help="Dataset to convert to YOLO format."
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the YOLO formatted dataset.",
    )
    args = parser.parse_args()

    # Check if the dataset is supported
    if args.dataset not in ["ipd"]:
        raise ValueError("Unsupported dataset: %s" % args.dataset)

    # Check if the output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Convert the dataset to YOLO format
    if args.dataset == "ipd":
        convert_ipd_to_yolo(args.output_dir)
    else:
        raise ValueError("Unsupported dataset: %s" % args.dataset)
