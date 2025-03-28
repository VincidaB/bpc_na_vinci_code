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

from cv_bridge import CvBridge
from geometry_msgs.msg import Pose as PoseMsg
from scipy.spatial.transform import Rotation

from ibpc_interfaces.msg import Camera as CameraMsg
from ibpc_interfaces.msg import Photoneo as PhotoneoMsg
from ibpc_interfaces.msg import PoseEstimate as PoseEstimateMsg
from ibpc_interfaces.srv import GetPoseEstimates
from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import Pose
import sys
import json

# This is the first version of the pipeline, using the following steps:
# YOLO11 --> FastSAM --> FoundationPose

# Done: - Load every model
# TODO: - Load images (from a split, or a scene of a split)\
# TODO: - compute the MSSD for each image

# TODO: - parametrable FoundationPose parameters


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
def ros_pose_to_mat(pose: PoseMsg):
    r = Rotation.from_quat(
        [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    )
    matrix = r.as_matrix()
    pose_matrix = np.eye(4)
    pose_matrix[:3, :3] = matrix
    pose_matrix[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    return pose_matrix


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

    def __init__(self, msg: Union[CameraMsg, PhotoneoMsg]):
        """
        Initializes a new Camera object from a ROS message.

        Args:
            msg: Either a CameraMsg or a PhotoneoMsg ROS message containing
                 camera information.

        Raises:
           TypeError: If the input `msg` is not of the expected type.
        """
        br = CvBridge()

        if not isinstance(msg, (CameraMsg, PhotoneoMsg)):
            raise TypeError("Input message must be of type CameraMsg or PhotoneoMsg")

        self.name: str = (msg.info.header.frame_id,)
        self.pose: np.ndarray = ros_pose_to_mat(msg.pose)
        self.intrinsics: np.ndarray = np.array(msg.info.k).reshape(3, 3)
        self.rgb = br.imgmsg_to_cv2(msg.rgb)
        self.depth = br.imgmsg_to_cv2(msg.depth)
        if isinstance(msg, CameraMsg):
            self.aolp: Optional[np.ndarray] = br.imgmsg_to_cv2(msg.aolp)
            self.dolp: Optional[np.ndarray] = br.imgmsg_to_cv2(msg.dolp)
        else:  # PhotoneoMsg
            self.aolp: Optional[np.ndarray] = None
            self.dolp: Optional[np.ndarray] = None


class pipeline_alpha:

    def __init__(
        self,
        detector_path: str,
        segmentor_path: str,
        resize_factor: float = 0.185,
        debug: int = 0,
    ):

        self.resize_factor = resize_factor
        self.debug = debug
        self.debug_dir = f"{os.path.dirname(__file__)}/debug"

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
    ) -> List[PoseEstimateMsg]:
        """
        Computes the pose estimates for the objects in the scene,
        returns the object id, the score or confidence, and the pose
        """
        pose_estimates = []

        for cam in [cam_1, cam_2, cam_3]:
            cam_k = scale_cam_k(cam.intrinsics, self.resize_factor)
            color = cam.rgb
            depth = cam.depth  # Convert depth to float32
            # cv2.imshow("depth", depth)
            # cv2.waitKey(0)
            # TODO : decide on what aprt of the pipiline gets the scaled down image

            # 2D detection
            results = self.detector(color)
            boxes = results[0].cpu().boxes.numpy().xyxy

            # Segmentation
            everything_results = self.segmentor_predictor(color)

            bbox_results = self.segmentor_predictor.prompt(
                everything_results, bboxes=boxes
            )

            # save a mask image for each objects
            masks = self.generate_masks(bbox_results)
            color = resize(color, self.resize_factor)
            depth = resize(depth, self.resize_factor).astype(
                np.float32
            )  # Ensure depth remains float32 after resizing

            # TODO: we need to get the object id from the detection, currently we are using a 2D detector that only predicts obj_18, so we will use that for now
            # TODO: change this to load model from dataset path, maybe even the eval one
            mesh_file = f"{code_dir}/sixD_pose_estimation/FoundationPose/demo_data/ipd_val_0/mesh/obj_000018.obj"
            print(f"Loading mesh from {mesh_file}")
            mesh = trimesh.load(mesh_file)
            mesh.apply_transform(
                np.diag([0.001, 0.001, 0.001, 1])
            )  #! annnoying to remember, this should be baked in one of the method calls
            self.est.set_model(mesh.vertices, mesh.vertex_normals, mesh=mesh)
            for mask in masks:
                pose = self.est.register(
                    K=cam_k, rgb=color, depth=depth, ob_mask=mask, iteration=3
                )
                estimated_position = pose[:3, 3]
                estimated_rotation = pose[:3, :3]
                pose_msg = Pose()
                pose_msg.position.x = float(estimated_position[0])
                pose_msg.position.y = float(estimated_position[1])
                pose_msg.position.z = float(estimated_position[2])
                r = Rotation.from_matrix(estimated_rotation)
                quat = r.as_quat()
                pose_msg.orientation.x = quat[0]
                pose_msg.orientation.y = quat[1]
                pose_msg.orientation.z = quat[2]
                pose_msg.orientation.w = quat[3]

                pose_estimate = PoseEstimateMsg(obj_id=18, score=0.8, pose=pose_msg)
                pose_estimates.append(pose_estimate)

                print(f"Estimated position: {estimated_position}")
                print(f"Estimated rotation: {estimated_rotation}")

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

    # testing the pipeline_alpha class
    pipeline = pipeline_alpha(
        detector_path="./bpc/2D_detection/yolo11_ipd/yolov11m_ipd_train_on_test/weights/best.pt",
        segmentor_path="./bpc/segmentation/FastSAM/weights/FastSAM-x.pt",
        resize_factor=0.185,
        debug=0,
    )

    # let's Create 3 cameras and photeneo = None
    # we will use the same image for all cameras for now
    code_dir = os.path.dirname(__file__)
    dataset = f"{code_dir}/../datasets/ipd"
    split = "val"
    scene_id = 0
    img_id = 0
    color = cv2.imread(image_path(dataset, split, scene_id, "rgb", "cam1", img_id))
    depth = cv2.imread(
        image_path(dataset, split, scene_id, "depth", "cam1", img_id),
        cv2.IMREAD_UNCHANGED,
    )
    aolp = cv2.imread(image_path(dataset, split, scene_id, "aolp", "cam1", img_id), -1)
    dolp = cv2.imread(image_path(dataset, split, scene_id, "dolp", "cam1", img_id), -1)
    depth = depth.astype(np.float32)
    depth = depth / 10000

    with open(camera_json_path(dataset, split, scene_id, "cam1"), "r") as f:
        camera_data = json.load(f)
    cam_k = np.array(camera_data[str(img_id)]["cam_K"]).reshape(3, 3)
    print("Camera Intrinsics Matrix (cam_K):")
    print(cam_k)

    # Use the actual cam_k matrix for CameraMsg
    camera_info = CameraInfo()
    camera_info.header.frame_id = "camera_1"
    camera_info.k = cam_k.flatten().tolist()

    # TODO : get the actual pose of the camera (extrinsics)
    pose = Pose()
    pose.position.x = 0.0
    pose.position.y = 0.0
    pose.position.z = 0.0
    pose.orientation.x = 0.0
    pose.orientation.y = 0.0
    pose.orientation.z = 0.0
    pose.orientation.w = 1.0

    br = CvBridge()

    rgb_image = br.cv2_to_imgmsg(color, encoding="8UC3")
    depth_image = br.cv2_to_imgmsg(depth, encoding="32FC1")
    aolp_image = br.cv2_to_imgmsg(aolp, encoding="8UC1")
    dolp_image = br.cv2_to_imgmsg(dolp, encoding="8UC1")

    # Create CameraMsg, but this should be called scene, or cameras_msg
    camera_msg = CameraMsg(
        info=camera_info,
        pose=pose,
        rgb=rgb_image,
        depth=depth_image,
        aolp=aolp_image,
        dolp=dolp_image,
    )

    cam_1 = Camera(camera_msg)

    start_time_pose_estimation = time.time()
    poses = pipeline.get_pose_estimates(
        object_ids=[18],
        cam_1=cam_1,
        cam_2=cam_1,
        cam_3=cam_1,
        photoneo=None,
    )
    logging.info(
        f"\033[32mPose estimation time: {time.time() - start_time_pose_estimation:.2f} seconds\033[0m"
    )

    print(poses)
