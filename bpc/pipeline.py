from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from typing import List
import cv2
import numpy as np
import os


# just a test to see if I can subscribe to a topic from a conda env

# subribe to the topic /chatter for now

from pipeline_alpha import pipeline_alpha
from pipeline_alpha import Camera

# from ibpc_interfaces.msg import Camera as CameraMsg
# from ibpc_interfaces.msg import Photoneo as PhotoneoMsg
# from ibpc_interfaces.msg import PoseEstimate as PoseEstimateMsg

code_dir = os.path.dirname(__file__)

pipeline = pipeline_alpha(
    detector_path=f"{code_dir}/2D_detection/yolo11_ipd/yolov11m_ipd_train_on_test/weights/best.pt",
    segmentor_path=f"{code_dir}/segmentation/FastSAM/weights/FastSAM-x.pt",
    resize_factor=0.185,
    debug=0,
)


app = FastAPI()


class DepthPayload(BaseModel):
    data: List[int]  # Flattened depth array
    width: int  # Width of the depth image
    height: int


class PoseRequest(BaseModel):
    object_ids: List[int]
    cam_1: bytes  # send JPEG or PNG-encoded bytes
    cam_1_depth: DepthPayload  # Use the new depth payload structure
    cam_1_intrinsics: List[float]
    cam_1_extrinsics: List[float]


@app.post("/estimate")
def estimate_pose(req: PoseRequest):
    import base64

    # Decode RGB image
    image_bytes = base64.b64decode(req.cam_1)
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode RGB image. Ensure the input is valid.")

    # Decode depth image
    depth_data = np.array(req.cam_1_depth.data, dtype=np.uint16)
    depth = depth_data.reshape((req.cam_1_depth.height, req.cam_1_depth.width))
    # Check if depth image is valid
    print("max depth value:", np.max(depth))
    print("min depth value:", np.min(depth))

    if depth is None:
        raise ValueError("Failed to decode depth image. Ensure the input is valid.")

    # Debugging: Display the images
    # cv2.imshow("RGB Image", img)
    # cv2.imshow("Depth Image", depth)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    cam_1 = Camera(
        frame_id="cam_1",
        pose=np.array(req.cam_1_extrinsics).reshape((4, 4)),
        intrinsics=np.array(req.cam_1_intrinsics).reshape((3, 3)),
        rgb=img,
        depth=depth,
    )

    poses = pipeline.get_pose_estimates(
        object_ids=req.object_ids,
        cam_1=cam_1,
        cam_2=cam_1,
        cam_3=cam_1,
        photoneo=None,
    )

    return {"poses": poses}  # Ensure poses is JSON serializable

    cam_1 = Camera(
        frame_id="cam_1",
        pose=np.array(req.cam_1_extrinsics).reshape((4, 4)),
        intrinsics=np.array(req.cam_1_intrinsics).reshape((3, 3)),
        rgb=img,
        depth=depth,
    )

    poses = pipeline.get_pose_estimates(
        object_ids=[18],
        cam_1=cam_1,
        cam_2=cam_1,
        cam_3=cam_1,
        photoneo=None,
    )

    poses = []  # TODO : Replace with actual pose estimation logic

    return {"poses": poses}  # make sure poses is JSON serializable


if __name__ == "__main__":
    # start the pipeline
    uvicorn.run(app, host="127.0.0.1", port=8000)
