# Datasets directory Structure

Datasets are expeceted to be store in this directory (`datasets/`).

## Structure

``` bash
datasets/
├── idp/  # dataset direcory
│   ├── ipd_models/
│   |   ├── models/
|   │   │   ├── models_info.json
|   │   │   ├── obj_000000.ply
|   │   │   ├── ... # obj 8, 10, 11, 14, 18, 19, 20
|   │   │   ├── obj_000020.ply
│   |   ├── models_eval/ # much bigger models for some reason
|   │   │   ├── models_info.json
|   │   │   ├── obj_000000.ply
|   │   │   ├── ... # obj 8, 10, 11, 14, 18, 19, 20
│   ├── train_pbr/
│   │   ├── 000000/
│   │   │   ├── aolp_cam1/
│   │   │   ├── depth_cam1/
│   │   │   ├── dolp_cam1/
│   │   │   ├── mask_cam1/
│   │   │   ├── mask_visib_cam1/
│   │   │   ├── rgb_cam1/
│   │   │   ├── scene_camera_cam1.json
│   │   │   ├── scene_gt_cam1.json
│   │   │   ├── scene_gt_info_cam1.json
│   │   │   ├── ... # cam 2, 3
│   │   ├── ...
│   │   ├── 000009/
│   ├── val/
│   │   ├── 000000/
│   │   │   ├── aolp_cam1/
│   │   │   ├── depth_cam1/
│   │   │   ├── depth_photoeno/
│   │   │   ├── dolp_cam1/
│   │   │   ├── mask_cam1/
│   │   │   ├── mask_visib_cam1/
│   │   │   ├── rgb_cam1/
│   │   │   ├── rgb_photoeno/
│   │   │   ├── scene_camera_cam1.json
│   │   │   ├── scene_gt_cam1.json
│   │   │   ├── scene_gt_info_cam1.json
│   │   │   ├── ... # cam 2, 3
│   │   ├── ...
│   │   ├── 000014/
│   ├── test/
│   │   ├── 000000/
│   │   │   ├── aolp_cam1/
│   │   │   ├── depth_cam1/
│   │   │   ├── dolp_cam1/
│   │   │   ├── mask_cam1/
│   │   │   ├── mask_visib_cam1/
│   │   │   ├── rgb_cam1/
│   │   │   ├── scene_camera_cam1.json
│   │   │   ├── ... # cam 2, 3
│   │   ├── ...
│   │   ├── 000014/
├── yolo11/
│   ├── ipd_bop_data_jan25_1_obj_11/
│   │   ├── images/
│   │   ├── labels/
```

## Description
- `idp/` is the dataset directoy.
  - `ipd_models/` 
    - `models/` contains `.ply` files representing 3D object models.
    - `models_eval/` contains `.ply` files representing 3D object models.
  - `train_pbr/` holds multiple scenes, each containing different camera captures and metadata JSON files, with ground truth poses.
  - `val/` holds multiple scenes, each containing different camera captures and metadata JSON files, with ground truth poses.
  - `test/` holds multiple scenes, each containing different camera captures and metadata JSON files, **without** ground truth poses.
- `yolo_ipd/` contains YOLO-formatted datasets:
  - `yolo_ipd.yaml` is the dataset configuration file.
  - `train/`:
    - `images/` holds training images.
    - `labels/` stores corresponding YOLO labels.
  - `val/`:
    - `images/` holds validation images.
    - `labels/` stores corresponding YOLO labels.
  - `test/`:
    - `images/` holds test images.

The `yolo_ipd` directory is the YOLO formatted version of the IPD dataset. It is used for training and testing YOLO models. it can be generated using the `bpc/2D_detection/bop_to_yolo_dataset.py` script.

### Usage
```bash
python bpc/2D_detection/bop_to_yolo_dataset.py --dataset ipd --output_dir yolo_ipd
```


## Tips

To not copy your whole `ipd` dataset in the `datasets` directory, you can create a symbolic link to the `ipd` dataset directory.

```bash
ln -s /path/to/your/ipd/dataset datasets/ipd
```

We also use this trick to avoid copying the `yolo_ipd` dataset in the `datasets` directory.
