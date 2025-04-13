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

root_path = "/media/vincent/more/bpc_teamname/bpc"

pipeline = pipeline_alpha(
    detector_path=f"{root_path}/2D_detection/yolo11_ipd/yolov11m_ipd_train_on_test/weights/best.pt",
    segmentor_path=f"{root_path}/segmentation/FastSAM/weights/FastSAM-x.pt",
    resize_factor=0.185,
    debug=0,
)

code_dir = os.path.dirname(__file__)


app = FastAPI()


class PoseRequest(BaseModel):
    object_ids: List[int]
    cam_1: bytes  # send JPEG or PNG-encoded bytes
    cam_1_depth: bytes  # send JPEG or PNG-encoded bytes
    cam_1_intrinsics: List[float]


# def image_decoder(image_bytes: bytes) -> np.ndarray:
#     # Decode base64 string to bytes
#     nparr = np.frombuffer(image_bytes, np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#     return img


@app.post("/estimate")
def estimate_pose(req: PoseRequest):
    # print("Received request:", req)
    # Decode image
    import base64

    # Decode base64 string to bytes
    image_bytes = base64.b64decode(req.cam_1)
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # show the image for debug
    if img is None:
        raise ValueError("Failed to decode image. Ensure the input is valid.")

    depth_bytes = base64.b64decode(req.cam_1_depth)
    nparr_depth = np.frombuffer(depth_bytes, np.uint8)
    depth = cv2.imdecode(nparr_depth, cv2.IMREAD_UNCHANGED)

    if False:
        cv2.imshow("image", img)
        cv2.waitKey(0)

    cam_1 = Camera(
        frame_id="cam_1",
        pose=np.eye(4),
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
