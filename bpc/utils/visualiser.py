import open3d as o3d
import cv2
import numpy as np
import json
import open3d.visualization.gui as gui
import open3d.t.geometry as tgeometry
import random
import argparse
import time
from scipy.spatial.transform import Rotation
import pickle
import torch

modalities = ["rgb", "depth"]
cameras = ["cam1", "cam2", "cam3"]


def image_path(
    split: str, scene_id: int, image_id: int, modality: str, camera: str
) -> str:
    if split == "train_pbr" and modality != "depth":
        ext = "jpg"
    else:
        ext = "png"
    return f"{dataset_path}/{split}/{scene_id:06d}/{modality}_{camera}/{image_id:06d}.{ext}"


def camera_json_path(split: str, scene_id: int, camera: str) -> str:
    return f"{dataset_path}/{split}/{scene_id:06d}/scene_camera_{camera}.json"


def camera_gt_json_path(split: str, scene_id: int, camera: str) -> str:
    return f"{dataset_path}/{split}/{scene_id:06d}/scene_gt_{camera}.json"


def camera_gt_info_json_path(split: str, scene_id: int, camera: str) -> str:
    return f"{dataset_path}/{split}/{scene_id:06d}/scene_gt_info_{camera}.json"


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


def get_camera_gt(
    split: str, scene_id: int, image_id: int, camera: str
) -> tuple[list[float], list[float]]:
    with open(camera_gt_json_path(split, scene_id, camera)) as f:
        data = json.load(f)

    return data[str(image_id)]


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
    trans /= -1000.0  # convert to meters
    # return q, trans
    return r_inv.as_matrix(), trans


class camera_pov:
    # TODO : camera extrinsics need to be converted to m and changed to the correct frame

    def __init__(
        self,
        camera_intrinsics,
        camera_extrinsics,
        rgb_image,
        depth_image,
        resize_factor=1.0,
    ):
        self.camera_intrinsics = camera_intrinsics
        self.camera_extrinsics = camera_extrinsics
        self.rgb_image = rgb_image
        self.depth_image = depth_image

        self.point_cloud = None
        self.false_color = False

        if resize_factor != 1.0:
            self.depth_image = cv2.resize(
                depth_image, (0, 0), fx=resize_factor, fy=resize_factor
            )
            self.rgb_image = cv2.resize(
                rgb_image, (0, 0), fx=resize_factor, fy=resize_factor
            )
            print(self.camera_intrinsics)
            self.camera_intrinsics[:-1] = [
                i * resize_factor for i in camera_intrinsics[:-1]
            ]
            print("Resized camera intrinsics")
            print(self.camera_intrinsics)
        self.resize_factor = resize_factor

        # TODO : get the depth factor from the json file
        self.depth_factor = 1000.0 * 10.0

    def get_point_cloud(self) -> tgeometry.PointCloud:
        if self.point_cloud != None:
            return self.point_cloud

        depth = self.depth_image.astype(float) / self.depth_factor
        rows, cols = depth.shape
        c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
        Z = depth
        X = (c - self.camera_intrinsics[2]) * Z / self.camera_intrinsics[0]
        Y = (r - self.camera_intrinsics[5]) * Z / self.camera_intrinsics[4]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.dstack((X, Y, Z)).reshape(-1, 3))
        pcd.colors = o3d.utility.Vector3dVector(self.rgb_image.reshape(-1, 3) / 255.0)
        self.point_cloud = tgeometry.PointCloud.from_legacy(pcd)
        return self.point_cloud


def detections_to_position(
    detection: list, cameras: list[camera_pov]
) -> list[np.ndarray]:
    """
    converts the 2D detection to a 3D position in the camera frame
    returns a list of len the number of cameras, each element is a list of 3D points in the camera frame
    """

    if len(detection) != len(cameras):
        raise ValueError("Number of cameras and detections do not match")

    positions = []

    for i, camera in enumerate(cameras):
        positions.append([])
        K = np.array(camera.camera_intrinsics).reshape(3, 3)
        K_inv = np.linalg.inv(K)
        depth_image = camera.depth_image.astype(float)
        for box in results[i].boxes:
            # point_2 is the 2D detection (in pixels)
            point_2 = np.array(box.xywh.cpu())[0, :2]
            point_2 = point_2 * RESIZE_FACTOR

            # distance away from the camera using the depth image
            d = depth_image[int(point_2[1]), int(point_2[0])] / camera.depth_factor
            # project the 2D detection to a point in 3D space
            pixel_homog = np.array([point_2[0], point_2[1], 1.0])
            point_2_3D_cam_frame = (K_inv @ pixel_homog) * d
            positions[i].append(point_2_3D_cam_frame)
    print(positions)
    return positions


class PointCloudVisualizer:
    def __init__(self):
        self.coloured = False
        self.lit = False

        self.pov_cams = {}
        self.models = {}  # models in the scene

        gui.Application.instance.initialize()
        self.window = gui.Application.instance.create_window(
            "Point cloud camera pov viewer", 1920, 1080
        )

        self.scene = gui.SceneWidget()
        self.scene.scene = o3d.visualization.rendering.Open3DScene(self.window.renderer)

        self.scene.scene.set_background([1, 1, 1, 1])
        self.scene.scene.scene.set_sun_light(
            [-1, -1, -1], [1, 1, 1], 100000
        )  # direction # color # intensity
        self.scene.scene.scene.enable_sun_light(True)
        bbox = o3d.geometry.AxisAlignedBoundingBox([-1, -1, -1], [1, 1, 1])
        self.scene.setup_camera(60, bbox, [0, 0, 0])
        self.scene.scene.camera.look_at([0, 0, 0], [0, 0, -1], [0, -1, 0])
        self.window.add_child(self.scene)

        em = self.window.theme.font_size
        separation_height = int(round(0.5 * em))
        self._settings_panel = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em)
        )

        view_ctrls = gui.CollapsableVert(
            "View controls", 0.25 * em, gui.Margins(em, 0, 0, 0)
        )

        self._color_clouds_button = gui.Button("Toggle false colors")
        self._color_clouds_button.horizontal_padding_em = 0.5
        self._color_clouds_button.vertical_padding_em = 0
        self._color_clouds_button.set_on_clicked(
            lambda: self.change_point_clouds_color()
        )

        self._lit_unlit_button = gui.Button("Lit / Unlit")
        self._lit_unlit_button.horizontal_padding_em = 0.5
        self._lit_unlit_button.vertical_padding_em = 0
        self._lit_unlit_button.set_on_clicked(self.set_lit_unlit)

        self._icp_button = gui.Button("ICP")
        self._icp_button.horizontal_padding_em = 0.5
        self._icp_button.vertical_padding_em = 0
        self._icp_button.set_on_clicked(self.icp)

        view_ctrls.add_child(gui.Label("Controls"))
        h = gui.Horiz(0.25 * em)  # row 1
        h.add_stretch()
        h.add_child(self._color_clouds_button)
        h.add_child(self._lit_unlit_button)
        h.add_child(self._icp_button)
        h.add_stretch()
        view_ctrls.add_child(h)

        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(view_ctrls)

        self._settings_panel.visible = True

        self.window.add_child(self._settings_panel)
        self.window.set_on_layout(self._on_layout)

        app_menu = gui.Menu()
        app_menu.add_item("quit", 1)
        menu = gui.Menu()
        menu.add_menu("Nothing to see here", app_menu)

        gui.Application.instance.menubar = menu

        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = "defaultUnlit"

        # add a cube at zero zero zero
        cube = o3d.geometry.TriangleMesh.create_box(0.05, 0.05, 0.05)
        cube_t = tgeometry.TriangleMesh.from_legacy(cube)
        cube_t.translate([0, 0, 0])
        material.base_color = [1, 0, 0, 1]
        self.scene.scene.add_geometry("cube", cube_t, material)

        self.scene.scene.show_axes(True)

    def add_point_cloud(self, pov_cam: camera_pov, name="point_cloud", pose=None):
        if self.scene.scene.has_geometry(name):
            name = name + "_1"
        print(f"Adding point cloud with name '{name}'")
        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = "defaultUnlit"
        if pose is not None:
            r = np.array(pose[0], dtype=np.float64)
            center = np.array([0, 0, 0], dtype=np.float64)
            pov_cam.get_point_cloud().rotate(r, center)
            pov_cam.get_point_cloud().translate(pose[1])
        self.scene.scene.add_geometry(name, pov_cam.get_point_cloud(), material)
        print(f"Added point cloud with name '{name}'")
        # adding the camera pov to the dictionary with it's name as the key
        self.pov_cams[name] = pov_cam

    def change_point_clouds_color(self):
        for name, pov_cam in self.pov_cams.items():

            if pov_cam.false_color:
                print("Changing back to the original color")
                pov_cam.false_color = False
                pov_cam.point_cloud.point["colors"] = o3d.core.Tensor(
                    pov_cam.rgb_image.reshape(-1, 3) / 255.0,
                    dtype=o3d.core.Dtype.Float32,
                )
            else:
                # use the name of the camera to determine the color
                np.random.seed(sum(ord(char) for char in name))
                color = np.random.rand(3)
                pov_cam.false_color = True
                pov_cam.point_cloud.point["colors"] = o3d.core.Tensor(
                    np.ones_like(pov_cam.rgb_image).reshape(-1, 3) * color,
                    dtype=o3d.core.Dtype.Float32,
                )
            if self.scene.scene.has_geometry(name):
                self.scene.scene.scene.update_geometry(
                    name,
                    pov_cam.point_cloud,
                    o3d.visualization.rendering.Scene.UPDATE_COLORS_FLAG,
                )
            else:
                print(f"No geometry with the name '{name}' found")

    def add_ply_mesh(self, path: str, name: str, pose=None):
        if self.scene.scene.has_geometry(name):
            name = name + "_1"
        print(f"Adding ply with name '{name}'")
        ply = o3d.io.read_triangle_mesh(path)
        # scale down to meters
        ply.scale(1 / 1000, center=[0, 0, 0])
        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = "defaultUnlit"
        np.random.seed(sum(ord(char) for char in name))
        material.base_color = np.append(np.random.rand(3), 1)
        if pose is not None:
            r = pose[0]
            ply.rotate(r, center=[0, 0, 0])
            ply.translate(pose[1])
        self.scene.scene.add_geometry(name, ply, material)
        self.models[name] = ply
        print(f"Added ply with name '{name}'")

    def set_lit_unlit(self):
        material = o3d.visualization.rendering.MaterialRecord()
        self.lit = not self.lit
        material.shader = "defaultLit" if self.lit else "defaultUnlit"
        self.scene.scene.update_material(material)

    def icp(self):
        print("Running ICP with one iteration between cam1 and cam2 (hardcoded)")
        source = self.pov_cams["cam1"].get_point_cloud().to_legacy()
        target = self.pov_cams["cam2"].get_point_cloud().to_legacy()
        target.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        threshold = 0.02
        reg_p2l = o3d.pipelines.registration.registration_icp(
            source,
            target,
            threshold,
            np.identity(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1),
        )

        print("ICP transformation:")
        print(reg_p2l.transformation)
        current_pose = self.scene.scene.get_geometry_transform("obj18")
        new_pose = np.matmul(current_pose, reg_p2l.transformation)
        self.scene.scene.set_geometry_transform(
            "obj18", new_pose
        )  # https://www.open3d.org/docs/release/python_api/open3d.visualization.rendering.Open3DScene.html
        print(type(source))

    def add_object_in_camera_frame(
        self, object: o3d.geometry, pose, camera_name: str, material, name: str
    ):
        """
        What do I want this function to do ?
        - take a geometry object that has not been translated and rotated and place it in the camera frame ?
        - `pose` is the pose of the object in the camera frame
        - `camera_name` is the name of the camera
        """
        camera = self.pov_cams[camera_name]
        # get the camera pose

        Rc, Tc = camera_pose_from_extrinsics(
            camera.camera_extrinsics[0], camera.camera_extrinsics[1]
        )
        r, t = pose
        # change the object to the camera frame
        position = Rc @ t + Tc
        object.translate(position)

        # rotate the object around it's center
        rot_obj = Rc @ r
        object.rotate(rot_obj, center=position)

        if self.scene.scene.has_geometry(name):
            name = name + "_1"
            while self.scene.scene.has_geometry(name):
                # increase the number at the end
                name = name[:-1] + str(int(name[-1]) + 1)
        self.scene.scene.add_geometry(name, object, material)

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        self.scene.frame = r
        width = 25 * layout_context.theme.font_size
        height = min(
            r.height,
            self._settings_panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()
            ).height,
        )
        self._settings_panel.frame = gui.Rect(r.get_right() - width, r.y, width, height)

    def run(self):

        # could also use run_in_thread, cf https://www.open3d.org/docs/latest/python_api/open3d.visualization.gui.Application.html#open3d.visualization.gui.Application
        while gui.Application.instance.run_one_tick():
            time.sleep(1 / 60)  # 60 fps


if __name__ == "__main__":
    print(
        "██████╗  ██████╗ ██╗███╗   ██╗████████╗     ██████╗██╗      ██████╗ ██╗   ██╗██████╗\n██╔══██╗██╔═══██╗██║████╗  ██║╚══██╔══╝    ██╔════╝██║     ██╔═══██╗██║   ██║██╔══██╗\n██████╔╝██║   ██║██║██╔██╗ ██║   ██║       ██║     ██║     ██║   ██║██║   ██║██║  ██║\n██╔═══╝ ██║   ██║██║██║╚██╗██║   ██║       ██║     ██║     ██║   ██║██║   ██║██║  ██║\n██║     ╚██████╔╝██║██║ ╚████║   ██║       ╚██████╗███████╗╚██████╔╝╚██████╔╝██████╔╝\n╚═╝      ╚═════╝ ╚═╝╚═╝  ╚═══╝   ╚═╝        ╚═════╝╚══════╝ ╚═════╝  ╚═════╝ ╚═════╝\n\n██╗   ██╗██╗███████╗██╗    ██╗███████╗██████╗\n██║   ██║██║██╔════╝██║    ██║██╔════╝██╔══██╗\n██║   ██║██║█████╗  ██║ █╗ ██║█████╗  ██████╔╝\n╚██╗ ██╔╝██║██╔══╝  ██║███╗██║██╔══╝  ██╔══██╗\n ╚████╔╝ ██║███████╗╚███╔███╔╝███████╗██║  ██║\n  ╚═══╝  ╚═╝╚══════╝ ╚══╝╚══╝ ╚══════╝╚═╝  ╚═╝"
    )
    parser = argparse.ArgumentParser(description="Point Cloud Visualizer")
    parser.add_argument("dataset_path", type=str, help="Path to the dataset")
    parser.add_argument("--scene-id", type=int, default=0, help="Scene ID (default: 0)")
    parser.add_argument("--image-id", type=int, default=0, help="Image ID (default: 0)")
    parser.add_argument(
        "--split", type=str, default="test", help="Dataset split (default: test)"
    )
    parser.add_argument(
        "--resize-factor",
        type=float,
        default=1.0,
        help="Resize factor for images (default: 1.0)",
    )
    args = parser.parse_args()

    dataset_path = args.dataset_path
    scene_id = args.scene_id
    img_id = args.image_id
    split = args.split
    RESIZE_FACTOR = args.resize_factor

    rgb_images = [
        cv2.imread(image_path(split, scene_id, img_id, "rgb", camera))
        for camera in cameras
    ]
    depth_images = [
        cv2.imread(
            image_path(split, scene_id, img_id, "depth", camera), cv2.IMREAD_UNCHANGED
        )
        for camera in cameras
    ]

    camera_intrinsics = [
        get_camera_intrinsic(split, scene_id, img_id, camera) for camera in cameras
    ]
    camera_extrinsics = [
        get_camera_extrinsics(split, scene_id, img_id, camera) for camera in cameras
    ]

    visualizer = PointCloudVisualizer()
    pov_cam1 = camera_pov(
        camera_intrinsics[0],
        camera_extrinsics[0],
        rgb_images[0],
        depth_images[0],
        resize_factor=RESIZE_FACTOR,
    )
    pov_cam2 = camera_pov(
        camera_intrinsics[1],
        camera_extrinsics[1],
        rgb_images[1],
        depth_images[1],
        resize_factor=RESIZE_FACTOR,
    )
    pov_cam3 = camera_pov(
        camera_intrinsics[2],
        camera_extrinsics[2],
        rgb_images[2],
        depth_images[2],
        resize_factor=RESIZE_FACTOR,
    )

    visualizer.add_point_cloud(
        pov_cam1,
        "cam1",
        camera_pose_from_extrinsics(camera_extrinsics[0][0], camera_extrinsics[0][1]),
    )
    visualizer.add_point_cloud(
        pov_cam2,
        "cam2",
        camera_pose_from_extrinsics(camera_extrinsics[1][0], camera_extrinsics[1][1]),
    )
    visualizer.add_point_cloud(
        pov_cam3,
        "cam3",
        camera_pose_from_extrinsics(camera_extrinsics[2][0], camera_extrinsics[2][1]),
    )

    # used only to place a random object in the scene
    rot = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

    # TODO add option to add ply in the frame of of the cameras, currently it is in the world frame
    # visualizer.add_ply_mesh(
    #    f"{dataset_path}/ipd_models/models/obj_000018.ply",
    #    "obj18",
    #    pose=(rot, [-0.25, -0.1, 1.75]),
    # )

    # open pickle file of the 2D detection results, and lets draw lines from the camera to the 2D detections for now
    results_file_path = (
        "/media/vincent/more/bpc_teamname/bpc/2D_detection/test000000_000000.pkl"
    )
    results_file_path = (
        "/media/vincent/more/bpc_teamname/bpc/2D_detection/val000000_000000.pkl"
    )
    with open(results_file_path, "rb") as f:
        results = pickle.load(f)

    det_pos_cam_frame = detections_to_position(results, [pov_cam1, pov_cam2, pov_cam3])

    colors = [[1, 0, 0, 0.75], [0, 1, 0, 0.75], [0, 0, 1, 0.75]]
    for i in range(len(det_pos_cam_frame)):
        for j in range(len(det_pos_cam_frame[i])):
            red_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.025)
            red_sphere_t = tgeometry.TriangleMesh.from_legacy(red_sphere)
            material = o3d.visualization.rendering.MaterialRecord()

            # I would really like this to be `Unlit` but it is crashing open3D
            material.shader = "defaultLitTransparency"
            material.base_color = colors[i]
            visualizer.add_object_in_camera_frame(
                red_sphere_t,
                (np.eye(3), det_pos_cam_frame[i][j]),
                f"cam{i+1}",
                material,
                name=f"detection",
            )

    # lets load the ground truth 3D poses and draw them in the scene
    img_info = get_camera_gt(split, scene_id, img_id, "cam1")
    print(img_info)
    print("len(img_info) :")
    print(len(img_info))

    for obj_gt in img_info:
        rot = np.array(obj_gt["cam_R_m2c"]).reshape(3, 3)
        trans = np.array(obj_gt["cam_t_m2c"]) / 1000

        obj_18 = o3d.io.read_triangle_mesh(
            f"{dataset_path}/ipd_models/models/obj_000018.ply"
        )
        obj_18.scale(1 / 1000, center=[0, 0, 0])
        material = o3d.visualization.rendering.MaterialRecord()

        material.shader = "defaultUnlit"
        material.base_color = colors[0]
        visualizer.add_object_in_camera_frame(
            obj_18,
            (rot, trans),
            "cam1",
            material,
            name=f"detection",
        )
        # first, lets not care about rotation

    # TODO : look at using threads to run actions while the visualizer is running, look at the ICP example
    visualizer.run()
