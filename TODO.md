TODO soon: 


- [ ] handle multiple 2D detectors
    - [ ] store multiple 2D detectors
    - [x] select only the relevant 2D detectors depending on obj_id --> kind of urgent, we need to only use thew 2D detector of the requested object, we are currently loosing time in the 2D detection step for no reason
    - [x] URGENT: fix the ibpc_pose_estimator or whatever is halucinating or reading out of range of the returned pose estimations : we are getting poses read somewhere even when we are not populating the pose estimations. Actually, we are even running FoundationPose when we should not...
    It was on the ibpc_pose_estimator side of things :
- [x] load the different 3D models depending on the object detected. --> loading `.ply` works !
- [ ] check that the refiner is not using the image with the green bounding boxes, as it could hinder the pose estimation
- [x] return the correct pose of the object in the world frame
- [x] go through only 1 or 2 or 3 cameras
- [ ] Check the results :
    - [ ] check the 2D detections
    - [ ] check the masks
    - [ ] check the 3D estimated poses
- [ ] Pourquoi on passe autant de temps sur le fait de charger un nouveau mesh dans FoundationPose, on peut pas garder une partie des infos en cache ? 

- [ ] change `requirements.txt` to reflect the new packages used in the conda env

```bash
docker buildx build -t navincicode \
                                               --file ./Dockerfile.estimator \
                                               .
```
```bash
docker run --rm -it --gpus all navincicode:latest
```
Not tested with stupid `rocker`



5 iterations :

without symetries being taken into account : 
========================================
Error in position: 0.0012m
Angle est/gt rotation: 1.90 deg
========================================
[<module>()] register time second run: 2.54
[<module>()] register time third run: 2.60


with symetries being taken into account :
========================================
Error in position: 0.0013m
Angle est/gt rotation: 1.58 deg
========================================
[<module>()] register time second run: 1.29
[<module>()] register time third run: 1.30



-------------------------------
3 iterations :

without symetries being taken into account :
========================================
Error in position: 0.0012m
Angle est/gt rotation: 1.53 deg
========================================
[<module>()] register time second run: 1.88
[<module>()] register time third run: 1.78

with symetries being taken into account :
========================================
Error in position: 0.0012m
Angle est/gt rotation: 1.53 deg
========================================
[<module>()] register time second run: 0.90
[<module>()] register time third run: 0.90




30 min on the split `val` with symmetries NOT being taken into account 

22 min on the split `val` with symmetries taken into account