from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from typing import List
import cv2
import numpy as np

# just a test to see if I can subscribe to a topic from a conda env

# subribe to the topic /chatter for now


app = FastAPI()


#! pipeline = pipeline_alpha() 

class PoseRequest(BaseModel):
    object_ids: List[int]
    cam_1: bytes  # send JPEG or PNG-encoded bytes


@app.post("/estimate")
def estimate_pose(req: PoseRequest):
    print("Received request:", req)
    # Decode image
    import base64
    # Decode base64 string to bytes
    image_bytes = base64.b64decode(req.cam_1)
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # show the image for debug
    if img is None:
        raise ValueError("Failed to decode image. Ensure the input is valid.")
    
    if False:
        cv2.imshow("image", img)
        cv2.waitKey(0)
    
    # poses = pipeline.get_pose_estimates(
    #     object_ids=req.object_ids,
    #     cam_1=cam,
    #     cam_2=cam,
    #     cam_3=cam,
    #     photoneo=None,
    # )

    poses = []  # TODO : Replace with actual pose estimation logic

    return {"poses": poses}  # make sure poses is JSON serializable

if __name__ == '__main__':
    # start the pipeline
    uvicorn.run(app, host="127.0.0.1", port=8000)
