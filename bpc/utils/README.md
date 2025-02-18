## Utils

Here are some utility scripts that were used during the development of the project.

### visualiser.py

Used to display point clouds formed by the depth images of multiple cameras. The point clouds can be false colored based on the camera they belong to, in order to distinguish between them. 

<p float="left">
  <img src="../../assets/images/pc_viewer.png" alt="Point cloud viewer interface" width="45%" />
  <img src="../../assets/images/pv_viewer_false_color.png" alt="Point cloud viewer interface but the point clouds are false colored" width="45%" />
</p>

ICP can be used to align the point clouds, even though we will use the camera extrinsics to align them in the final implementation.