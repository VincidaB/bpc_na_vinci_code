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
import base64

import time

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

        image2 = request.cameras[1].rgb
        image2 = bridge.imgmsg_to_cv2(image2, desired_encoding="passthrough")

        image3 = request.cameras[2].rgb
        image3 = bridge.imgmsg_to_cv2(image3, desired_encoding="passthrough")

        depth = request.cameras[0].depth
        print("encoding: ", depth.encoding)  # --> 32FC1

        # works but I dont want the scaling, I want to keep the depth values
        # depth = bridge.imgmsg_to_cv2(depth, desired_encoding="passthrough")
        # depth = (depth * 255 / np.max(depth)).astype(np.uint8)  # Normalize and convert to 8-bit

        depth = bridge.imgmsg_to_cv2(depth, desired_encoding="32FC1")

        depth_array = np.array(depth, dtype=np.int32)
        print("depth array: ", depth_array)
        print("max depth: ", np.max(depth))
        print("min depth: ", np.min(depth))

        # Save the depth image to a file for debugging purposes
        # cv2.imwrite("/tmp/depth_debug.png", depth)
        # self.get_logger().info("Depth image saved to /tmp/depth_debug.png")

        # depth = cv2.imread('/media/vincent/more/bpc_teamname/000000_depth_cam1.png')

        # send the iamge as an array

        if image is None:
            self.get_logger().error("Failed to load image. Exiting...")
            return response

        image_bytes = cv2.imencode(".png", image)[1].tobytes()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        image2_bytes = cv2.imencode(".png", image2)[1].tobytes()
        image2_base64 = base64.b64encode(image2_bytes).decode("utf-8")

        image3_bytes = cv2.imencode(".png", image3)[1].tobytes()
        image3_base64 = base64.b64encode(image3_bytes).decode("utf-8")

        # depth_bytes = cv2.imencode(".png", depth)[1].tobytes()
        # depth_base64 = base64.b64encode(depth_bytes).decode("utf-8")

        depth_payload = {
            "data": depth_array.flatten().tolist(),
            "width": depth_array.shape[1],
            "height": depth_array.shape[0],
        }

        depth_payload2 = {
            "data": depth_array.flatten().tolist(),
            "width": depth_array.shape[1],
            "height": depth_array.shape[0],
        }
        depth_payload3 = {
            "data": depth_array.flatten().tolist(),
            "width": depth_array.shape[1],
            "height": depth_array.shape[0],
        }

        if not request.object_ids:
            self.get_logger().warn("Received request with empty object_ids.")
            return response

        intrinsics = np.array(request.cameras[0].info.k).tolist()
        camera_pose = ros_pose_to_mat(request.cameras[0].pose).flatten().tolist()
        intrinsics2 = np.array(request.cameras[1].info.k).tolist()
        camera_pose2 = ros_pose_to_mat(request.cameras[1].pose).flatten().tolist()
        intrinsics3 = np.array(request.cameras[2].info.k).tolist()
        camera_pose3 = ros_pose_to_mat(request.cameras[2].pose).flatten().tolist()

        self.get_logger().info("Camera intrinsics: " + str(intrinsics))
        self.get_logger().info("Camera extrinsics: " + str(camera_pose))

        object_ids = request.object_ids.tolist()

        payload = {
            "object_ids": object_ids,
            "cam_1": image_base64,
            "cam_1_depth": depth_payload,
            "cam_1_intrinsics": intrinsics,
            "cam_1_extrinsics": camera_pose,
            "cam_2": image2_base64,
            "cam_2_depth": depth_payload2,
            "cam_2_intrinsics": intrinsics2,
            "cam_2_extrinsics": camera_pose2,
            "cam_3": image3_base64,
            "cam_3_depth": depth_payload3,
            "cam_3_intrinsics": intrinsics3,
            "cam_3_extrinsics": camera_pose3,
        }
        max_retries = 5
        retry_delay = 2  # seconds

        for attempt in range(max_retries):
            try:
                res = requests.post("http://127.0.0.1:8000/estimate", json=payload)
                if res.ok:
                    break
                else:
                    self.get_logger().warn(
                        f"Attempt {attempt + 1} failed with status code {res.status_code}. Retrying..."
                    )
            except requests.exceptions.RequestException as e:
                self.get_logger().error(f"Request failed: {e}. Retrying...")
                time.sleep(retry_delay)
        else:
            self.get_logger().error(
                "Max retries reached. Unable to get a response from the server."
            )
            return response
        print("Response status code: ", res.status_code)
        print("Response content: ", res.content)

        response_data = json.loads(res.content.decode("utf-8"))

        pose_estimate_msg_list = []
        for pose in response_data["poses"]:
            pose_estimate_msg = PoseEstimateMsg()
            pose_estimate_msg.obj_id = pose["object_id"]
            pose_estimate_msg.pose.position.x = pose["pose"][0]
            pose_estimate_msg.pose.position.y = pose["pose"][1]
            pose_estimate_msg.pose.position.z = pose["pose"][2]
            pose_estimate_msg.pose.orientation.x = pose["pose"][3]
            pose_estimate_msg.pose.orientation.y = pose["pose"][4]
            pose_estimate_msg.pose.orientation.z = pose["pose"][5]
            pose_estimate_msg.pose.orientation.w = pose["pose"][6]
            pose_estimate_msg.score = pose["conf"]

            # Add the new message to the list
            pose_estimate_msg_list.append(pose_estimate_msg)

        if res.ok:
            print("Pose server response: " + res.text)
            response.pose_estimates = pose_estimate_msg_list
        else:
            self.get_logger().error("Pose server error: " + res.text)
            response.pose_estimates = pose_estimate_msg_list

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
