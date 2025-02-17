import open3d as o3d
import cv2
import numpy as np
import json
import open3d.visualization.gui as gui
import open3d.t.geometry as tgeometry
import random
import argparse

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

    def get_point_cloud(self) -> tgeometry.PointCloud:
        if self.point_cloud != None:
            return self.point_cloud

        depth_factor = 10000.0

        depth = self.depth_image.astype(float) / depth_factor
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


class PointCloudVisualizer:
    def __init__(self):
        self.coloured = False
        self.lit = False

        self.pov_cams = {}

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

        view_ctrls.add_child(gui.Label("Controls"))
        h = gui.Horiz(0.25 * em)  # row 1
        h.add_stretch()
        h.add_child(self._color_clouds_button)
        h.add_child(self._lit_unlit_button)
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

    def add_point_cloud(self, pov_cam: camera_pov, name="point_cloud"):
        if self.scene.scene.has_geometry(name):
            name = name + "_1"
        print(f"Adding point cloud with name '{name}'")
        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = "defaultUnlit"
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

    def set_lit_unlit(self):
        material = o3d.visualization.rendering.MaterialRecord()
        self.lit = not self.lit
        material.shader = "defaultLit" if self.lit else "defaultUnlit"
        self.scene.scene.update_material(material)

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
        gui.Application.instance.run()


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
    visualizer.add_point_cloud(pov_cam1, "cam_1")
    visualizer.add_point_cloud(pov_cam2, "cam_2")
    visualizer.add_point_cloud(pov_cam3, "cam_3")
    # TODO : look at using threads to run actions while the visualizer is running, look at the ICP example
    visualizer.run()
