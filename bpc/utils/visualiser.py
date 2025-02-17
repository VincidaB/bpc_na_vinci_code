import open3d as o3d
import cv2
import numpy as np
import json
import open3d.visualization.gui as gui
import open3d.t.geometry as tgeometry

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

camera_intrinsics = [
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
cam_k = camera_intrinsics[0]
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


pcd_t = tgeometry.PointCloud.from_legacy(pcd)


class PointCloudVisualizer:
    def __init__(self):
        self.coloured = False
        self.lit = True

        gui.Application.instance.initialize()
        self.window = gui.Application.instance.create_window("Open3D", 1920, 1080)

        self.scene = gui.SceneWidget()
        self.scene.scene = o3d.visualization.rendering.Open3DScene(self.window.renderer)

        self.scene.scene.set_background([1, 1, 1, 1])
        self.scene.scene.scene.set_sun_light(
            [-1, -1, -1], [1, 1, 1], 100000  # direction  # color
        )  # intensity
        self.scene.scene.scene.enable_sun_light(True)
        bbox = o3d.geometry.AxisAlignedBoundingBox([-10, -10, -10], [10, 10, 10])
        self.scene.setup_camera(60, bbox, [0, 0, 0])
        self.window.add_child(self.scene)

        em = self.window.theme.font_size
        separation_height = int(round(0.5 * em))
        self._settings_panel = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em)
        )

        view_ctrls = gui.CollapsableVert(
            "View controls", 0.25 * em, gui.Margins(em, 0, 0, 0)
        )

        self._color_clouds_button = gui.Button("Toggle cloud RGB coloring")
        self._color_clouds_button.horizontal_padding_em = 0.5
        self._color_clouds_button.vertical_padding_em = 0
        self._color_clouds_button.set_on_clicked(self.change_point_clouds_color)

        self._lit_unlit_button = gui.Button("Lit / Unlit")
        self._lit_unlit_button.horizontal_padding_em = 0.5
        self._lit_unlit_button.vertical_padding_em = 0
        self._lit_unlit_button.set_on_clicked(self.set_lit_unlit)

        view_ctrls.add_child(gui.Label("Mouse controls"))
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
        menu.add_menu("app", app_menu)

        gui.Application.instance.menubar = menu

        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = "defaultLit"

        self.scene.scene.add_geometry("point_cloud", pcd_t, material)

    def change_point_clouds_color(self):
        if self.coloured:
            print("Changing back to the original color")
            self.coloured = False
            pcd_t.point["colors"] = o3d.core.Tensor(
                rgb_image.reshape(-1, 3) / 255.0, dtype=o3d.core.Dtype.Float32
            )
        else:
            self.coloured = True
            pcd_t.point["colors"] = o3d.core.Tensor(
                np.ones_like(rgb_image).reshape(-1, 3) * [0, 1, 0],
                dtype=o3d.core.Dtype.Float32,
            )
        if self.scene.scene.has_geometry("point_cloud"):
            self.scene.scene.scene.update_geometry(
                "point_cloud",
                pcd_t,
                o3d.visualization.rendering.Scene.UPDATE_COLORS_FLAG,
            )
        else:
            print("No geometry found")

    def set_lit_unlit(self):
        material = o3d.visualization.rendering.MaterialRecord()
        if self.lit:
            material.shader = "defaultUnlit"
            self.scene.scene.update_material(material)
            self.lit = False
        else:
            material.shader = "defaultLit"
            self.scene.scene.update_material(material)
            self.lit = True

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
    visualizer = PointCloudVisualizer()
    visualizer.run()
