import open3d as o3d
import cv2
import numpy as np
import json

dataset_path = "/media/vincent/more/bpc_teamname/datasets/ipd"
split = "test"
scene_id = 0
img_id = 0

modalities = ["rgb", "depth"]
cameras = ["cam1", "cam2", "cam3"]


def image_path(
    split: str, scene_id: int, image_id: int, modality: str, camera: str
) -> str:
    if split == "train_pbr":
        ext = "jpg"
    else:
        ext = "png"
    return f"{dataset_path}/{split}/{scene_id:06d}/{modality}_{camera}/{image_id:06d}.{ext}"


def camera_json_path(split: str, scene_id: int, camera: str) -> str:
    return f"{dataset_path}/{split}/{scene_id:06d}/scene_camera_{camera}.json"


def get_camera_intrinsic(
    split: str, scene_id: int, image_id: int, camera: str
) -> list[float]:
    with open(camera_json_path(split, scene_id, camera)) as f:
        data = json.load(f)
    return data[str(image_id)]["cam_K"]


def get_camera_extrinsics(
    split: str, scene_id: int, image_id: int, camera: str
) -> tuple[list[float], list[float]]:
    with open(camera_json_path(split, scene_id, camera)) as f:
        data = json.load(f)

    return data[str(image_id)]["cam_R_w2c"], data[str(image_id)]["cam_t_w2c"]


rgb_images = [
    cv2.imread(image_path(split, scene_id, img_id, "rgb", camera)) for camera in cameras
]
depth_images = [
    cv2.imread(
        image_path(split, scene_id, img_id, "depth", camera), cv2.IMREAD_UNCHANGED
    )
    for camera in cameras
]

caemra_intrinsics = [
    get_camera_intrinsic(split, scene_id, img_id, camera) for camera in cameras
]

# create a point cloud from depth images, for now just from the first camera
depth_image = depth_images[0]
rgb_image = rgb_images[0]

RESIZE_FACTOR = 1.0

depth_image = cv2.resize(depth_image, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
rgb_image = cv2.resize(rgb_image, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)

# camera intrinsics
# create a point cloud from depth image
cam_k = [
    3981.985991142684,
    0.0,
    1954.1872863769531,
    0.0,
    3981.985991142684,
    1103.6978149414062,
    0.0,
    0.0,
    1.0,
]

fx = cam_k[0] * RESIZE_FACTOR
fy = cam_k[4] * RESIZE_FACTOR
cx = cam_k[2] * RESIZE_FACTOR
cy = cam_k[5] * RESIZE_FACTOR

print(fx, fy, cx, cy)

factor = 10000.0

depth = depth_image.astype(float) / factor
rows, cols = depth.shape
c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
Z = depth
X = (c - cx) * Z / fx
Y = (r - cy) * Z / fy

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.dstack((X, Y, Z)).reshape(-1, 3))
pcd.colors = o3d.utility.Vector3dVector(rgb_image.reshape(-1, 3) / 255.0)
o3d.visualization.draw_geometries([pcd])
