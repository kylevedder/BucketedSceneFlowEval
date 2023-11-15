# File system assumptions

### Argoverse 2 Sensor Dataset

Somewhere on disk, have an `argoverse2/` folder so that the downloaded files live inside

```
argoverse2/train
argoverse2/val
argoverse2/test
```

and generate the train and val supervision labels to

```
argoverse2/train_sceneflow
argoverse2/val_sceneflow
```


The [Argoverse 2 Scene Flow generation script](https://github.com/kylevedder/argoverse2-sf) to compute ground truth flows for both `train/` and `val/`.

Please note that when downloaded from the cloud, these files may have a different top level directory format (their stored format keeps changing); you can solve this by moving the files or symlinking the appropriate directories into a different tree. We have uploaded [a prebuilt DockerHub image](https://hub.docker.com/repository/docker/kylevedder/argoverse2_sf/general) for running the generation script.

### Argoverse 2 Tiny Demo Dataset

To get started, we provide a directly downloadable [tiny demo dataset](https://github.com/kylevedder/zeroflow/files/13059582/argoverse2_tiny.zip) (5.5MB).

`argoverse2_tiny` contains three subfolders:

 - `argoverse2_tiny/val`: a single sequence with the single frame pair
 - `argoverse2_tiny/val_sceneflow`: the supervised ground truth for this frame pair
 - `argoverse2_tiny/val_supervised_out`: the output of the forward pass of [FastFlow3D, a supervised scene flow estimator](http://vedder.io/zeroflow). 
 
### Waymo Open

Download Waymo Open v1.4.2 (earlier versions lack map information) and the scene Flow labels contributed by _Scalable Scene Flow from Point Clouds in the Real World_ from the [Waymo Open download page](https://waymo.com/open/). We preprocess these files, both to convert them from an annoying proto file format to a standard Python format and to remove the ground points.

Do this using 

1. `data_prep_scripts/waymo/rasterize_heightmap.py` -- generate heightmaps in a separate folder used for ground removal
2. `data_prep_scripts/waymo/extract_flow_and_remove_ground.py` -- extracts the points into a pickle format and removes the groundplane using the generated heightmaps

We have uploaded [a prebuilt DockerHub image](https://hub.docker.com/repository/docker/kylevedder/zeroflow_waymo/general) for running the Waymo conversion scripts.
