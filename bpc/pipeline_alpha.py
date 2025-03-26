import os, sys
import argparse
import argcomplete

from ultralytics import YOLO

# from segmentation.FastSAM.fastsam import FastSAM, FastSAMPrompt
from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPredictor
import cv2
import numpy as np

# This is the first version of the pipeline, using the following steps:
# YOLO11 --> FastSAM --> FoundationPose

# TODO: - Load every model
# TODO: - Load images (from a split, or a scene of a split)\
# TODO: - compute the MSSD for each image

# TODO: - Way to change the YOLO11 model to a different one
# TODO: - Way to change the FastSAM model to a different one
# TODO: - parametrable FoundationPose parameters

twoD_detector = None
segmentor = None


def main():
    parser = argparse.ArgumentParser()
    argcomplete.autocomplete(parser)

    parser.add_argument(
        "--detector",
        type=str,
        default="./bpc/2D_detection/yolo11_ipd/yolov11m_ipd_train_on_test/weights/best.pt",
        help="Model to use for the 2D detector",
    )
    parser.add_argument(
        "--segmentor",
        type=str,
        default="./bpc/segmentation/FastSAM/weights/FastSAM-x.pt",
        help="Model to use for the segmentation",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="./dataset/ipd",
        help="Path to dataset to use for the evaluation",
    )
    parser.add_argument(
        "--split", type=str, default="val", help="Split to use for the evaluation"
    )
    parser.add_argument(
        "--scenes",
        type=str,
        default="0",
        help="Scenes to use for the evaluation, scene number or 'all'",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="./output_alpha",
        help="Output path to save the results",
    )

    parser.add_argument(
        "--sequential",
        type=bool,
        default=False,
        help="Weather to treat the images sequentially or not",
    )

    # make sure the models (files) exists
    args = parser.parse_args()
    if not os.path.exists(args.detector):
        print(f"File {args.detector} does not exist")
        sys.exit(1)
    if not os.path.exists(args.segmentor):
        print(f"File {args.segmentor} does not exist")
        sys.exit(1)

    # Load the models
    global twoD_detector
    global segmentor

    twoD_detector = YOLO(args.detector)
    segmentor = FastSAM(args.segmentor)

    img_path = "./000000_test_scaled_1920.png"

    # testing the detector
    results = twoD_detector(img_path)
    # results[0].show()

    # testing the segmentor
    # everything_results = segmentor("./000000_val_scaled_1920.png", device="cuda", retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)
    # everything_results[0].show(boxes=False, color_mode='instance')

    overrides = dict(
        task="segment",
        mode="predict",
        model=args.segmentor,
        save=False,
        imgsz=1024,
        conf=0.4,
        iou=0.9,
    )
    predictor = FastSAMPredictor(overrides=overrides)
    everything_results = predictor(img_path)
    # prompting the segmentor using boxes from the 2D detector

    boxes = results[0].cpu().boxes.numpy().xyxy
    print(boxes)

    bbox_results = predictor.prompt(everything_results, bboxes=boxes)
    bbox_results[0].show(boxes=False, color_mode="instance")

    # save a mask image for each objects
    for i, mask in enumerate(bbox_results[0].masks):
        mask_data = (
            mask.xy
        )  # A list of numpy arrays, where each array contains the [x, y] pixel coordinates for a single segmentation mask. Each array has shape (N, 2), where N is the number of points in the segment.
        img_shape = mask.orig_shape

        # convert the mask_data to integer values
        mask_data = [mask.astype(int) for mask in mask_data]

        print(mask_data)
        # draw a white mask on a black image
        mask_img = np.zeros((img_shape), dtype=np.uint8)
        cv2.fillPoly(mask_img, mask_data, 255)
        # scale to 720x405
        mask_img = cv2.resize(mask_img, (720, 405))
        cv2.imwrite(f"mask_{i}.png", mask_img)


if __name__ == "__main__":
    main()
