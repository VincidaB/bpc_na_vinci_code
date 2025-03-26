import os, sys
import argparse
import argcomplete

from ultralytics import YOLO
import time

# from segmentation.FastSAM.fastsam import FastSAM, FastSAMPrompt
from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPredictor
import cv2
import numpy as np

sys.path.append(
    os.path.join(os.path.dirname(__file__), "sixD_pose_estimation", "FoundationPose")
)

from sixD_pose_estimation.FoundationPose.estimater import *
from sixD_pose_estimation.FoundationPose.datareader import *

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


def image_path(
    dataset_path: str,
    split: str,
    scene_id: int,
    modality: str,
    camera: str,
    image_id: int,
) -> str:
    if split == "train_pbr" and modality != "depth":
        ext = "jpg"
    else:
        ext = "png"
    return f"{dataset_path}/{split}/{scene_id:06d}/{modality}_{camera}/{image_id:06d}.{ext}"


def main():
    parser = argparse.ArgumentParser()
    argcomplete.autocomplete(parser)

    code_dir = os.path.dirname(os.path.realpath(__file__))

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
        default="./datasets/ipd",
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
        "--resize-factor",
        type=float,
        default=0.25,
        help="Resize factor for images (default: 0.25)",
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
    parser.add_argument(
        "--est_refine_iter",
        type=int,
        default=5,
        help="Number of iterations for the pose refinement of FoundationPose",
    )
    parser.add_argument(
        "--debug",
        type=int,
        default=0,
        help="Debug level for FoundationPose",
    )
    parser.add_argument("--debug_dir", type=str, default=f"{code_dir}/debug")

    args = parser.parse_args()

    debug = args.debug
    debug_dir = args.debug_dir
    os.system(
        f"rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam"
    )

    set_logging_format()
    set_seed(0)

    resize_factor = args.resize_factor
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()

    # TODO : Load the mesh later depending on the object we detected
    #! never forget to scale the mesh down to meters
    mesh_file = f"{code_dir}/sixD_pose_estimation/FoundationPose/demo_data/ipd_val_0/mesh/obj_000018.obj"
    mesh = trimesh.load(mesh_file)
    mesh.apply_transform(np.diag([0.001, 0.001, 0.001, 1]))

    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=debug_dir,
        debug=debug,
        glctx=glctx,
    )

    # for visualizing the 3D model in debug mode
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    # make sure the models (files) exists
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

    dataset = args.dataset
    split = args.split
    scenes = args.scenes

    img_id = 0
    if scenes == "all":
        scenes = os.listdir(f"{dataset}/{split}")
    else:
        scenes = [int(scenes)]

    for scene_id in scenes:
        print(
            f"img path : ", image_path(dataset, split, scene_id, "rgb_cam", "1", img_id)
        )

    img_path = image_path(dataset, split, scene_id, "rgb", "cam1", img_id)

    results = twoD_detector(img_path)

    overrides = dict(
        task="segment",
        mode="predict",
        model=args.segmentor,
        save=False,
        imgsz=1024,
        conf=0.4,
        iou=0.8,
    )
    predictor = FastSAMPredictor(overrides=overrides)
    everything_results = predictor(img_path)
    # prompting the segmentor using boxes from the 2D detector

    boxes = results[0].cpu().boxes.numpy().xyxy
    print(boxes)

    bbox_results = predictor.prompt(everything_results, bboxes=boxes)
    bbox_results[0].show(boxes=False, color_mode="instance")

    masks = []

    # save a mask image for each objects
    for i, mask in enumerate(bbox_results[0].masks):
        mask_data = mask.xy
        img_shape = mask.orig_shape

        # convert the mask_data to integer values
        mask_data = [mask.astype(int) for mask in mask_data]

        # draw a white mask on a black image
        mask_img = np.zeros((img_shape), dtype=np.uint8)
        cv2.fillPoly(mask_img, mask_data, 255)
        original_shape = mask_img.shape
        mask_img = cv2.resize(
            mask_img,
            (
                int(original_shape[1] * resize_factor),
                int(original_shape[0] * resize_factor),
            ),
        )
        cv2.imwrite(f"mask_{i}.png", mask_img)

        masks.append(mask_img)

    # testing FoundationPose

    color = cv2.imread(img_path)
    color = cv2.resize(
        color,
        (int(color.shape[1] * resize_factor), int(color.shape[0] * resize_factor)),
    )

    depth = cv2.imread(
        image_path(dataset, split, scene_id, "depth", "cam1", img_id),
        cv2.IMREAD_UNCHANGED,
    )
    depth = cv2.resize(
        depth,
        (int(depth.shape[1] * resize_factor), int(depth.shape[0] * resize_factor)),
    )
    depth = depth.astype(np.float32)
    # scale depth by a factor of 0.1
    depth = depth / 10000

    # TODO : remove this loading of the mask with the file
    mask = cv2.imread("mask_4.png", -1)

    cam_k = np.array(
        [
            [746.6, 0.0, 366.4101161956787],
            [0.0, 746.64, 206.94334030151367],
            [0.0, 0.0, 1.0],
        ]
    )

    start_time = time.time()
    pose = est.register(
        K=cam_k, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter
    )
    logging.info(f"\033[33mregister time: {time.time() - start_time:.2f}\033[0m")

    estimated_position = pose[:3, 3]
    estimated_rotation = pose[:3, :3]
    logging.info(f"Estimated position: {estimated_position}")
    logging.info(f"Estimated rotation: {estimated_rotation}")

    os.makedirs(f"{debug_dir}/ob_in_cam", exist_ok=True)
    np.savetxt(f"{debug_dir}/ob_in_cam/000000.txt", pose.reshape(4, 4))
    # taken directly from the FoundationPose code
    if debug >= 1:
        center_pose = pose @ np.linalg.inv(to_origin)
        vis = draw_posed_3d_box(cam_k, img=color, ob_in_cam=center_pose, bbox=bbox)
        vis = draw_xyz_axis(
            color,
            ob_in_cam=center_pose,
            scale=0.1,
            K=cam_k,
            thickness=3,
            transparency=0,
            is_input_rgb=True,
        )

        cv2.imshow("3D Visualization", vis[..., ::-1])
        cv2.imshow("Mask", mask)
        cv2.waitKey(-1)

    if debug >= 2:
        os.makedirs(f"{debug_dir}/track_vis", exist_ok=True)
        imageio.imwrite(f"{debug_dir}/track_vis/000000.png", vis)


if __name__ == "__main__":
    main()
