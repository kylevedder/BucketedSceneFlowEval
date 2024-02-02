# File system assumptions

### Argoverse 2 Sensor Dataset

Somewhere on disk, have an `argoverse2/` folder so that the downloaded files live inside

```
argoverse2/train
argoverse2/val
argoverse2/test
```

Please note that when downloaded from the cloud, these files may have a different top level directory format (their stored format keeps changing); you can solve this by moving the files or symlinking the appropriate directories into a different tree.

Generate the train and val supervision labels to

```
argoverse2/train_sceneflow_feather
argoverse2/val_sceneflow_feather
```

To generate these supervision labels, use the generation script in `data_prep_scripts/argo/create_gt_flow.py`. We have uploaded [a prebuilt DockerHub image](https://hub.docker.com/repository/docker/kylevedder/zeroflow_av2/general) for running the generation script; it can be run using `./launch.sh`.

### Argoverse 2 Tiny Demo Dataset

To get started, we provide a directly downloadable [tiny demo dataset](https://github.com/kylevedder/BucketedSceneFlowEval/files/13881746/argoverse2_tiny.zip) (5.5MB).

`argoverse2_tiny` contains four subfolders:

 - `argoverse2_tiny/val`: a single sequence with the single frame pair
 - `argoverse2_tiny/val_sceneflow_feather`: the supervised ground truth for this frame pair
 - `argoverse2_tiny/val_nsfp_flow_feather`: the NSFP pseudolabels for this frame pair
 - `argoverse2_tiny/val_supervised_out`: the output of the forward pass of [FastFlow3D, a supervised scene flow estimator](http://vedder.io/zeroflow).

### Waymo Open

Download Waymo Open v1.4.2 (earlier versions lack map information) and the scene Flow labels contributed by _Scalable Scene Flow from Point Clouds in the Real World_ from the [Waymo Open download page](https://waymo.com/open/). We preprocess these files, both to convert them from an annoying proto file format to a standard Python format and to remove the ground points.

Do this using

1. `data_prep_scripts/waymo/rasterize_heightmap.py` -- generate heightmaps in a separate folder used for ground removal
2. `data_prep_scripts/waymo/extract_flow_and_remove_ground.py` -- extracts the points into a pickle format and removes the groundplane using the generated heightmaps

We have uploaded [a prebuilt DockerHub image](https://hub.docker.com/repository/docker/kylevedder/zeroflow_waymo/general) for running the Waymo conversion scripts.

### Waymo Open Tiny Demo Dataset

We have also provided a directly downloadable [tiny demo dataset](https://github.com/kylevedder/BucketedSceneFlowEval/files/13924555/waymo_open_processed_flow_tiny.zip) (3.1MB).

`waymo_open_processed_flow_tiny` contains two subfolders:

- `training`: a single frame pair of waymo data
- `train_nsfp_flow`: the flow labels for the framepair
