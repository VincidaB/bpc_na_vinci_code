# todo(Yadunund): Add copyright.

import cv2
from cv_bridge import CvBridge
import numpy as np
from scipy.spatial.transform import Rotation
import sys
from typing import List, Optional, Union

from geometry_msgs.msg import Pose as PoseMsg
from ibpc_interfaces.msg import Camera as CameraMsg
from ibpc_interfaces.msg import Photoneo as PhotoneoMsg
from ibpc_interfaces.msg import PoseEstimate as PoseEstimateMsg
from ibpc_interfaces.srv import GetPoseEstimates

import rclpy
from rclpy.node import Node
from pathlib import Path

import requests
import json

# Add the path three directories up to the system path
# Adjust the path to match the folder structure
# pipeline_alpha_path = Path(__file__).resolve().parents[3] / "pipeline_alpha"
# if pipeline_alpha_path.is_dir():
#     sys.path.append(str(pipeline_alpha_path))
# else:
#     raise FileNotFoundError(f"pipeline_alpha directory not found at {pipeline_alpha_path}")

# from pipeline_alpha.pipeline_alpha import pipeline_alpha

# pieline = pipeline_alpha(
#         detector_path="../../../bpc/2D_detection/yolo11_ipd/yolov11m_ipd_train_on_test/weights/best.pt",
#         segmentor_path="../../../bpc/segmentation/FastSAM/weights/FastSAM-x.pt",
#         resize_factor=0.185,
#         debug=0,
# )


# Helper functions
def ros_pose_to_mat(pose: PoseMsg):
    r = Rotation.from_quat(
        [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    )
    matrix = r.as_matrix()
    pose_matrix = np.eye(4)
    pose_matrix[:3, :3] = matrix
    pose_matrix[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    return pose_matrix


bridge = CvBridge()


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


class PoseEstimator(Node):

    def __init__(self):
        super().__init__("bpc_pose_estimator")
        self.get_logger().info("Starting bpc_pose_estimator...")
        # Declare parameters
        # self.model_dir = (
        #     self.declare_parameter("model_dir", "").get_parameter_value().string_value
        # )
        # if self.model_dir == "":
        #     raise Exception("ROS parameter model_dir not set.")
        # self.get_logger().info(f"Model directory set to {self.model_dir}.")
        srv_name = "/get_pose_estimates"
        self.get_logger().info(f"Pose estimates can be queried over srv {srv_name}.")
        self.srv = self.create_service(GetPoseEstimates, srv_name, self.srv_cb)

    def srv_cb(self, request, response):

        image = request.cameras[0].rgb
        image = bridge.imgmsg_to_cv2(image, desired_encoding="passthrough")

        depth = request.cameras[0].depth
        depth = bridge.imgmsg_to_cv2(depth, desired_encoding="passthrough")

        # image = cv2.imread('/media/vincent/more/bpc_teamname/mask_0.png')
        # depth = cv2.imread('/media/vincent/more/bpc_teamname/000000_depth_cam1.png')

        if image is None:
            self.get_logger().error("Failed to load image. Exiting...")
            return response

        import base64

        image_bytes = cv2.imencode(".png", image)[1].tobytes()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        depth_bytes = cv2.imencode(".png", depth)[1].tobytes()
        depth_base64 = base64.b64encode(depth_bytes).decode("utf-8")

        if not request.object_ids:
            self.get_logger().warn("Received request with empty object_ids.")
            return response

        intrinsics = np.array(request.cameras[0].info.k).tolist()
        self.get_logger().info("Camera intrinsics: " + str(intrinsics))

        payload = {
            "object_ids": [18],
            "cam_1": image_base64,
            "cam_1_depth": depth_base64,
            "cam_1_intrinsics": intrinsics,
        }

        res = requests.post("http://127.0.0.1:8000/estimate", json=payload)

        change_this = PoseEstimateMsg()

        if res.ok:
            print("Pose server response: " + res.text)
            response.pose_estimates = [change_this]
        else:
            self.get_logger().error("Pose server error: " + res.text)
            response.pose_estimates = [change_this]

        return response

    def get_pose_estimates(
        self,
        object_ids: List[int],
        cam_1: Camera,
        cam_2: Camera,
        cam_3: Camera,
        photoneo: Camera,
    ) -> List[PoseEstimateMsg]:

        pose_estimates = []
        print(f"Received request to estimates poses for object_ids: {object_ids}")
        """
            Your implementation goes here.
            msg = PoseEstimateMsg()
            # your logic here.
            pose_estimates.append(msg)
        """
        return pose_estimates


def main(argv=sys.argv):
    rclpy.init(args=argv)

    pose_estimator = PoseEstimator()

    rclpy.spin(pose_estimator)

    rclpy.shutdown()


if __name__ == "__main__":
    main()
