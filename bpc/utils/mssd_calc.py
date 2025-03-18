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
from copy import deepcopy


if __name__ == "__main__":
    # I want to implement MSSD calculation :
    # MSSD is Maximum Symetry-Aware Surface Distance

    MODEL = 18
    models_info_path = "../../datasets/ipd/ipd_models/models_eval/models_info.json"
    model_path = f"../../datasets/ipd/ipd_models/models_eval/obj_{MODEL:06d}.ply"

    with open(models_info_path, "r") as f:
        models_info = json.load(f)
    # print(models_info)

    model_info = models_info[str(MODEL)]
    # print(model_info)

    if (
        model_info["symmetries_continuous"] is None
        and model_info["symmetries_discrete"] is None
    ):
        print("No symmetries")

    # in mm for now, remember to scale down everything when implementing this in FoundationPose
    test_pose = (
        np.array([1, 2, 5]),
        np.array([1, 2, 5]),
    )  # Slight translation and rotation --> MSSD != 0 expected
    test_pose = (
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
    )  # No translation and rotation --> MSSD = 0 expected
    test_pose = (
        np.array([0, 0, 0]),
        np.array([180, 0, 0]),
    )  # 180 degrees rotation around x axis --> MSSD = 0 expected
    test_pose = (
        np.array([1, 2, 5]),
        np.array([180, 2, 5]),
    )  # Slight translation and rotation compared to 180 --> MSSD != 0 expected

    # Load the 3D model
    model = o3d.io.read_triangle_mesh(model_path)
    model.compute_vertex_normals()

    # Set the model color to green
    model.paint_uniform_color([0, 1, 0])  # RGB: Green

    # Create a second model with the test_pose
    model_transformed = o3d.io.read_triangle_mesh(model_path)
    model_transformed.compute_vertex_normals()

    # we make a copy before applying the test pose
    model_transformed_copy = deepcopy(model_transformed)

    # Apply the test_pose to the second model
    translation, rotation = test_pose
    rotation_matrix = Rotation.from_euler("xyz", rotation, degrees=True).as_matrix()
    model_transformed.translate(translation)
    model_transformed.rotate(rotation_matrix, center=(0, 0, 0))

    # Set the second model color to orange
    model_transformed.paint_uniform_color([1, 0.5, 0])  # RGB: Orange

    # Create a visualizer

    # let's try to compute MSSD between the two models
    # MSSD is Maximum Symetry-Aware Surface Distance
    # It is defined as such
    # ```math
    # e_{MSPD} \left( \hat{P}, \bar{P}, S_M, V_M \right) =
    # min_{S \in S_M} max_{x \in V_M}  \left| \left| \hat{P}x  - \bar{P}Sx  \right| \right|_2
    # ```

    # where:
    # - $\hat{P}$ is the estimated pose
    # - $\bar{P}$ is the ground truth pose
    # - $S_M$ is the set that contaains global symetry transformations of the object model $M$, cf "BOP challenge 2020 on 6D object localization" section 2.3
    # - $V_M$ is the set of vertices of the object model $M$

    # first, we compute the max distance for wach vertex without symetry

    max_d = 0
    for i in range(len(model.vertices)):
        d = np.linalg.norm(model.vertices[i] - model_transformed.vertices[i])
        if d > max_d:
            max_d = d

    print(f"Max distance without symetry: {max_d}")

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Add both models to the visualizer
    vis.add_geometry(model)
    vis.add_geometry(model_transformed)

    # let's only handle discrete symmetries for now
    for symetry in model_info["symmetries_discrete"]:
        rotation_matrix = np.array(symetry).reshape(4, 4)[
            :3, :3
        ]  # Extract 3x3 rotation matrix
        # first make a copy of the model
        # rotated_model = model_transformed.copy() # no worky
        rotated_model = model_transformed_copy
        # set color to red
        rotated_model.paint_uniform_color([1, 0, 0])

        rotated_model = rotated_model.rotate(rotation_matrix, center=(0, 0, 0))
        # Add the rotated model to the visualizer
        vis.add_geometry(rotated_model)

        max_d = 0
        for i in range(len(model.vertices)):
            d = np.linalg.norm(
                rotated_model.vertices[i] - model_transformed.vertices[i]
            )
            if d > max_d:
                max_d = d

        print(f"Max distance with symetry: {max_d}")

    # Set the initial view
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, -1])
    ctr.set_up([0, -1, 0])
    ctr.set_zoom(0.8)

    # Run the visualizer
    vis.run()
    vis.destroy_window()
