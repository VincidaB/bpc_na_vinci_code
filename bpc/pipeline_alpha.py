import os, sys
import argparse

from ultralytics import YOLO
import time

# from segmentation.FastSAM.fastsam import FastSAM, FastSAMPrompt
from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPredictor
import cv2
import numpy as np

# sys.path.append(
#     os.path.join(os.path.dirname(__file__), "sixD_pose_estimation", "FoundationPose")
# )

from sixD_pose_estimation.FoundationPose.estimater import *
from sixD_pose_estimation.FoundationPose.datareader import *

# from cv_bridge import CvBridge
# from geometry_msgs.msg import Pose as PoseMsg
from scipy.spatial.transform import Rotation

# from ibpc_interfaces.msg import Camera as CameraMsg
# from ibpc_interfaces.msg import Photoneo as PhotoneoMsg
# from ibpc_interfaces.msg import PoseEstimate as PoseEstimateMsg
# from ibpc_interfaces.srv import GetPoseEstimates
# from sensor_msgs.msg import CameraInfo, Image
# from geometry_msgs.msg import Pose
import sys
import json
import imageio

# This is the first version of the pipeline, using the following steps:
# YOLO11 --> FastSAM --> FoundationPose

# Done: - Load every model
# TODO: - Load images (from a split, or a scene of a split)\
# TODO: - compute the MSSD for each image

# TODO: - parametrable FoundationPose parameters

code_dir = os.path.dirname(__file__)


def camera_pose_from_extrinsics(
    R: list[float], t: list[float]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts the camera extrinsics to a pose and rotation in the world frame
    """
    r_mat = np.array(R).reshape(3, 3)
    r = Rotation.from_matrix(r_mat)
    r_inv = r.inv()
    # q = r_inv.as_quat()
    trans = r_mat.T @ np.array(t)
    # trans /= -1000.0  # convert to meters
    trans = -trans
    # return q, trans
    return r_inv.as_matrix(), trans


def resize(img, factor):
    return cv2.resize(
        img,
        (int(img.shape[1] * factor), int(img.shape[0] * factor)),
    )


def scale_cam_k(cam_k: np.ndarray, factor: float) -> np.ndarray:
    """
    Scales the camera intrinsics matrix by a given factor.

    Args:
        cam_k (np.ndarray): The original camera intrinsics matrix.
        factor (float): The scaling factor.

    Returns:
        np.ndarray: The scaled camera intrinsics matrix.
    """
    scaled_cam_k = cam_k.copy()
    scaled_cam_k[0, 0] *= factor
    scaled_cam_k[1, 1] *= factor
    scaled_cam_k[0, 2] *= factor
    scaled_cam_k[1, 2] *= factor
    return scaled_cam_k


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


def camera_json_path(dataset_path: str, split: str, scene_id: int, camera: str) -> str:
    return f"{dataset_path}/{split}/{scene_id:06d}/scene_camera_{camera}.json"


#         # Helper functions
# def ros_pose_to_mat(pose: PoseMsg):
#     print(f"Pose: {pose}")
#     # return np.eye(4)
#     r = Rotation.from_quat(
#         [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
#     )
#     matrix = r.as_matrix()
#     pose_matrix = np.eye(4)
#     pose_matrix[:3, :3] = matrix
#     pose_matrix[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
#     return pose_matrix


class Camera:
    """
    Represents a camera with its pose, intrinsics, and image data,
    initialized from either a CameraMsg or a PhotoneoMsg ROS message.

    Attributes:
        name (str): The name of the camera (taken from the message's frame_id).
        pose (np.ndarray): The 4x4 camera pose matrix (world-to-camera transformation),
                          converted from the ROS message's pose.
        intrinsics (np.ndarray): The 3x3 camera intrinsics matrix, reshaped from the
                                  message's K matrix.
        rgb (np.ndarray): The RGB image data, converted from the ROS message.
        depth (np.ndarray): The depth image data, converted from the ROS message.
        aolp (np.ndarray, optional): The Angle of Linear Polarization data,
                                     converted from the CameraMsg (None for PhotoneoMsg).
        dolp (np.ndarray, optional): The Degree of Linear Polarization data,
                                     converted from the CameraMsg (None for PhotoneoMsg).
    """

    def __init__(
        self,
        frame_id: str,
        pose: np.ndarray,
        intrinsics: np.ndarray,
        rgb,
        depth,
    ):
        print(f"Camera Pose: {pose}")
        self.name: str = (frame_id,)
        self.pose: np.ndarray = pose
        self.intrinsics: np.ndarray = intrinsics
        self.rgb = rgb
        self.depth = depth


class pipeline_alpha:

    def __init__(
        self,
        detector_path: str,
        segmentor_path: str,
        resize_factor: float = 0.185,
        debug: int = 3,
    ):

        self.resize_factor = resize_factor
        self.debug = debug
        self.debug_dir = f"{os.path.dirname(__file__)}/debug"
        # os.system(f'rm -rf {self.debug_dir}/* && mkdir -p {self.debug_dir}/track_vis {self.debug_dir}/ob_in_cam')
        print(f"debug dir: {self.debug_dir}")

        self.detector = YOLO(detector_path)
        print(f"Initializing segmentor with the model located at: {segmentor_path}")
        if not os.path.exists(segmentor_path):
            raise FileNotFoundError(f"Segmentor model not found at {segmentor_path}")
        self.segmentor = FastSAM(segmentor_path)

        self.overridess = dict(
            task="segment",
            mode="predict",
            model=segmentor_path,  #! this requires a model path
            save=False,
            imgsz=1024,
            conf=0.4,
            iou=0.8,
        )
        self.segmentor_predictor = FastSAMPredictor(overrides=self.overridess)

        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()
        self.glctx = dr.RasterizeCudaContext()
        self.est = FoundationPose(
            scorer=self.scorer,
            refiner=self.refiner,
            debug_dir=self.debug_dir,
            debug=self.debug,
            glctx=self.glctx,
        )

    def get_pose_estimates(
        self,
        object_ids: List[int],
        cam_1: Camera,
        cam_2: Camera,
        cam_3: Camera,
        photoneo: np.array,
    ):
        """
        Computes the pose estimates for the objects in the scene,
        returns the object id, the score or confidence, and the pose
        """
        pose_estimates = []
        cameras_to_use = [
            cam_1,
            # cam_2,
            # cam_3
        ]
        for cam in cameras_to_use:
            cam_k = scale_cam_k(cam.intrinsics, self.resize_factor)
            print(f"Camera intrinsics matrix (cam_K): {cam_k}")
            camera_pose = cam.pose

            color = cam.rgb
            depth = cam.depth

            # TODO : decide on what part of the pipeline gets the scaled down image

            # 2D detection
            results = self.detector(color)
            boxes = results[0].cpu().boxes.numpy().xyxy
            # TODO : check if the boxes are empty, we need to return an empty pose list in that case
            if len(boxes) == 0:
                print("No boxes detected")
                continue

            # Segmentation
            everything_results = self.segmentor_predictor(color)
            print(f"Found {len(everything_results[0].masks)} masks in the image")
            print(f"Found {len(results[0].boxes)} boxes in the image")
            bbox_results = self.segmentor_predictor.prompt(
                everything_results, bboxes=boxes
            )

            # save a mask image for each objects
            masks = self.generate_masks(bbox_results)
            color = resize(color, self.resize_factor)
            depth = resize(depth, self.resize_factor)

            # max value in depth :

            # Convert depth to float32 and scale down properly
            depth = depth.astype(np.float32)

            depth /= 10000

            # show depth using matplotlib
            # plt.imshow(depth)
            # plt.colorbar()
            # plt.show()

            # TODO: we need to get the object id from the detection, currently we are using a 2D detector that only predicts obj_18, so we will use that for now
            # TODO: change this to load model from dataset path, maybe even the eval one
            print(f"Object ids to detect : {object_ids}")
            for i, object_id in enumerate(object_ids):
                if object_id != 18:
                    print(
                        f"currently, only object 18 is supported, skipping {object_id}"
                    )
                    continue

                mesh_file = f"{code_dir}/../meshes/models/obj_{object_id:06d}.ply"
                # mesh_file = "/media/vincent/more/bpc_teamname/bpc/sixD_pose_estimation/FoundationPose/demo_data/ipd_val_0/mesh/obj_000018.obj"
                print(f"Loading mesh from {mesh_file}")
                mesh = trimesh.load(mesh_file)
                mesh.apply_transform(
                    np.diag([0.001, 0.001, 0.001, 1])
                )  #! annnoying to remember, this should be baked in one of the method calls
                self.est.set_model(mesh.vertices, mesh.vertex_normals, mesh=mesh)

                # for debug modes
                to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
                bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

                for j, mask in enumerate(masks):

                    # save color, depth and mask images in the debug directory
                    os.makedirs(f"{self.debug_dir}/mask", exist_ok=True)
                    os.makedirs(f"{self.debug_dir}/color", exist_ok=True)
                    os.makedirs(f"{self.debug_dir}/depth", exist_ok=True)

                    imageio.imwrite(
                        f"{self.debug_dir}/mask/{i}-{j}.png", mask.astype(np.uint8)
                    )
                    imageio.imwrite(
                        f"{self.debug_dir}/color/{i}-{j}.png", color.astype(np.uint8)
                    )
                    imageio.imwrite(
                        f"{self.debug_dir}/depth/{i}-{j}.png",
                        (depth * 10000).astype(np.uint16),
                    )

                    pose = self.est.register(
                        K=cam_k, rgb=color, depth=depth, ob_mask=mask, iteration=3
                    )
                    estimated_position = pose[:3, 3]
                    estimated_rotation = pose[:3, :3]
                    # pose_msg = Pose()

                    os.makedirs(f"{self.debug_dir}/ob_in_cam", exist_ok=True)
                    np.savetxt(
                        f"{self.debug_dir}/ob_in_cam/{i}-{j}.txt", pose.reshape(4, 4)
                    )

                    print(f"self.debug: {self.debug}")
                    # if self.debug >= 3:
                    #     print("self.debug>=3")
                    #     m = mesh.copy()
                    #     m.apply_transform(pose)
                    #     m.export(f"{self.debug_dir}/model_tf.obj")
                    #     xyz_map = depth2xyzmap(depth, cam_k)
                    #     valid = depth >= 0.001
                    #     pcd = toOpen3dCloud(xyz_map[valid], color[valid])
                    #     o3d.io.write_point_cloud(
                    #         f"{self.debug_dir}/scene_complete.ply", pcd
                    #     )

                    # if self.debug >= 1:
                    #     center_pose = pose @ np.linalg.inv(to_origin)
                    #     vis = draw_posed_3d_box(
                    #         cam_k, img=color, ob_in_cam=center_pose, bbox=bbox
                    #     )
                    #     vis = draw_xyz_axis(
                    #         color,
                    #         ob_in_cam=center_pose,
                    #         scale=0.1,
                    #         K=cam_k,
                    #         thickness=3,
                    #         transparency=0,
                    #         is_input_rgb=True,
                    #     )
                    #     cv2.imshow("1", vis[..., ::-1])
                    #     cv2.waitKey(10)

                    # if self.debug >= 2:
                    #     os.makedirs(f"{self.debug_dir}/track_vis", exist_ok=True)
                    #     imageio.imwrite(f"{self.debug_dir}/track_vis/{i}-{j}.png", vis)

                    #! change the confidence (maybe use the score from the detection)

                    print(f"Estimated position: {estimated_position}")
                    print(f"Estimated rotation: {estimated_rotation}")

                    # change the frame of the pose from the camera frame to the world frame

                    Rc, Tc = camera_pose_from_extrinsics(
                        camera_pose[:3, :3], camera_pose[:3, 3] / 1000
                    )

                    # Rc, Tc = camera_pose[:3, :3], camera_pose[:3, 3] /1000
                    # print(f"Rc: {Rc}")
                    # print(f"Tc: {Tc}")
                    world_position = Rc @ estimated_position + Tc
                    world_rotation = Rc @ estimated_rotation
                    # print(f"Estimated world position: {world_position}")
                    # print(f"Estimated world rotation: {world_rotation}")

                    r = Rotation.from_matrix(world_rotation)
                    quat = r.as_quat().tolist()
                    world_position = world_position.tolist()

                    pose_not_msg = (
                        world_position + quat
                    )  # concatenate the position and rotation

                    print(f"Pose but not a msg: {pose_not_msg}")
                    pose_estimate = {
                        "object_id": object_id,
                        "conf": 0.8,
                        "pose": pose_not_msg,
                    }
                    pose_estimates.append(pose_estimate)

        return pose_estimates

    def generate_masks(self, bbox_results):
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
            mask_img = resize(mask_img, self.resize_factor)
            masks.append(mask_img)
        return masks


if __name__ == "__main__":
    pass

    # # testing the pipeline_alpha class
    # pipeline = pipeline_alpha(
    #     detector_path="./bpc/2D_detection/yolo11_ipd/yolov11m_ipd_train_on_test/weights/best.pt",
    #     segmentor_path="./bpc/segmentation/FastSAM/weights/FastSAM-x.pt",
    #     resize_factor=0.185,
    #     debug=3,
    # )

    # # let's Create 3 cameras and photeneo = None
    # # we will use the same image for all cameras for now
    # dataset = f"{code_dir}/../datasets/ipd"
    # split = "val"
    # scene_id = 0
    # img_id = 0
    # color = cv2.imread(image_path(dataset, split, scene_id, "rgb", "cam1", img_id))
    # depth = cv2.imread(
    #     image_path(dataset, split, scene_id, "depth", "cam1", img_id),
    #     cv2.IMREAD_UNCHANGED,
    # )
    # aolp = cv2.imread(image_path(dataset, split, scene_id, "aolp", "cam1", img_id), -1)
    # dolp = cv2.imread(image_path(dataset, split, scene_id, "dolp", "cam1", img_id), -1)
    # depth = depth.astype(np.float32)
    # depth = depth / 10000

    # with open(camera_json_path(dataset, split, scene_id, "cam1"), "r") as f:
    #     camera_data = json.load(f)
    # cam_k = np.array(camera_data[str(img_id)]["cam_K"]).reshape(3, 3)
    # print("Camera Intrinsics Matrix (cam_K):")
    # print(cam_k)

    # # Use the actual cam_k matrix for CameraMsg
    # # camera_info = CameraInfo()
    # # camera_info.header.frame_id = "camera_1"
    # # camera_info.k = cam_k.flatten().tolist()

    # # TODO : get the actual pose of the camera (extrinsics)
    # # pose = Pose()
    # # pose.position.x = 0.0
    # # pose.position.y = 0.0
    # # pose.position.z = 0.0
    # # pose.orientation.x = 0.0
    # # pose.orientation.y = 0.0
    # # pose.orientation.z = 0.0
    # # pose.orientation.w = 1.0

    # #br = CvBridge()

    # #rgb_image = br.cv2_to_imgmsg(color, encoding="8UC3")
    # #depth_image = br.cv2_to_imgmsg(depth, encoding="32FC1")
    # #aolp_image = br.cv2_to_imgmsg(aolp, encoding="8UC1")
    # #dolp_image = br.cv2_to_imgmsg(dolp, encoding="8UC1")

    # cam_1 = Camera(
    #     frame_id="camera_1",
    #     pose=pose,
    #     intrinsics=cam_k,
    #     rgb=color,
    #     depth=depth,
    # )

    # start_time_pose_estimation = time.time()
    # poses = pipeline.get_pose_estimates(
    #     object_ids=[18],
    #     cam_1=cam_1,
    #     cam_2=cam_1,
    #     cam_3=cam_1,
    #     photoneo=None,
    # )
    # logging.info(
    #     f"\033[32mPose estimation time: {time.time() - start_time_pose_estimation:.2f} seconds\033[0m"
    # )

    # print(poses)
