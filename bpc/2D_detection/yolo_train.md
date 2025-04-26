# Train your YOLO models

## Directory structure

To train a model, make sure you have the necessary datasets. Instructions are given for the following repository structure. Datasets are available on [HuggingFace](https://huggingface.co/datasets/bop-benchmark/ipd).

```bash
bpc_na_vinci_code/
├── bpc/
│   ├── 2D_detection/
│   │   ├── train.py
├── datasets/
│   ├── data/
│   │   ├── models_eval/              # Evaluation models (resampled 3D models)
│   │   ├── models/                   # Original 3D object models
│   │   ├── train_pbr/                # Photorealistic synthetic training data
│   │   │   ├── 000000/
│   │   │   │   ├── scene_gt_info_cam1.json   # Ground truth info (camera 1)
│   │   │   │   ├── scene_gt_info_cam2.json   # Ground truth info (camera 2)
│   │   │   │   ├── scene_gt_info_cam3.json   # Ground truth info (camera 3)
│   │   │   │   ├── rgb_cam1/                 # RGB images (camera 1)
│   │   │   │   ├── depth_cam1/               # Depth images (camera 1)
│   │   │   │   ├── mask_cam1/                # Object segmentation masks (camera 1)
│   │   │   │   ├── rgb_cam2/                 # RGB images (camera 2)
│   │   │   │   ├── depth_cam2/               # Depth images (camera 2)
│   │   │   │   ├── mask_cam2/                # Object segmentation masks (camera 2)
│   │   │   │   ├── rgb_cam3/                 # RGB images (camera 3)
│   │   │   │   ├── depth_cam3/               # Depth images (camera 3)
│   │   │   │   ├── mask_cam3/                # Object segmentation masks (camera 3)
│   │   │   │   ├── scene_camera_cam1.json    # Camera parameters (cam1)
│   │   │   │   ├── scene_camera_cam2.json    # Camera parameters (cam2)
│   │   │   │   ├── scene_camera_cam3.json    # Camera parameters (cam3)
│   │   │   │   ├── scene_gt_cam1.json        # Object ground truth poses (cam1)
│   │   │   │   ├── scene_gt_cam2.json        # Object ground truth poses (cam2)
│   │   │   │   ├── scene_gt_cam3.json        # Object ground truth poses (cam3)

```

## Training your first model

To train a new model, you must first prepare the data. Starting from the root directory `bpc_na_vinci_code`, if you want to train the model for the object 18:
```bash
python3 bpc/2D_detection/prepare_data.py \
    --dataset_path "datasets/data/train_pbr" \
    --output_path "datasets/2D_detection/train_obj_18" \
    --obj_id 18
```

This should only take a few seconds. Then, you can start training. Still from the root, you can use:
```bash
python3 bpc/2D_detection/train.py \
    --obj_id 18 \
    --data_path "bpc/2D_detection/configs/data_obj_18.yaml" \
    --epochs 20 \
    --imgsz 640 \
    --batch 16 \
    --task detection

```
A full training will typically require more than 20 epochs.
In addition to information displayed in the terminal, you can monitor the training by launching from a second terminal, still from `bpc_na_vinci_code`:
```
tensorboard --logdir runs/detect/trainX
```
where `X` is the number associated to your training (for the very first run, there is no `X`). You can check the first lines of the terminal logs to find which `X` you're currently working with.

## Resuming a training 

After completing a full training, if you see that the metrics are still getting better in your validation set, it means you should probably resume training to get the best possible model. To do so:
```bash 
python3 bpc/2D_detection/train.py \
    --obj_id 18 \
    --data_path "bpc/2D_detection/configs/data_obj_18.yaml" \
    --epochs 40 \
    --imgsz 640 \
    --batch 16 \
    --task detection \
    --resume True \
    --model_path runs/detect/train/weights/best.pt
```

## Observing the results

In additions to the metrics, you should take a look at the images in `runs/detect/trainX`. You can also run:
```bash
python3 bpc/2D_detection/inference_sample_yolo.py \
    --model_path "bpc/yolo/models/detection/obj_18/yolo11-detection-obj_18.pt" \
    --image_path "datasets/data/train_pbr/000005/rgb_cam1/000001.jpg"
```