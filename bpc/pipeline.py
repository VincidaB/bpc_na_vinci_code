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

# detector_paths is now a dict where the key is the object id and the value is the path to the model
detectors = {
    0: f"{code_dir}/2D_detection/weights/obj_0/yolo11-detection-obj_0.pt",
    1: f"{code_dir}/2D_detection/weights/obj_1/yolo11-detection-obj_1.pt",
    4: f"{code_dir}/2D_detection/weights/obj_4/yolo11-detection-obj_4.pt",
    8: f"{code_dir}/2D_detection/weights/obj_8/yolo11-detection-obj_8.pt",
    10: f"{code_dir}/2D_detection/weights/obj_10/yolo11-detection-obj_10.pt",
    11: f"{code_dir}/2D_detection/weights/obj_11/yolo11-detection-obj_11.pt",
    14: f"{code_dir}/2D_detection/weights/obj_14/yolo11-detection-obj_14.pt",
    # 18: f"{code_dir}/2D_detection/yolo11_ipd/yolov11m_ipd_train_on_test/weights/best.pt",
    18: f"{code_dir}/2D_detection/weights/obj_18/yolo11-detection-obj_18.pt",
    19: f"{code_dir}/2D_detection/weights/obj_19/yolo11-detection-obj_19.pt",
    20: f"{code_dir}/2D_detection/weights/obj_20/yolo11-detection-obj_20.pt",
}


pipeline = pipeline_alpha(
    detector_paths=detectors,
    segmentor_path=f"{code_dir}/segmentation/FastSAM/weights/FastSAM-x.pt",
    resize_factor=0.37,
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


if __name__ == "__main__":
    # start the pipeline
    print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH"))
    uvicorn.run(app, host="127.0.0.1", port=8000)
