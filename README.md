# Bucketed Scene Flow Evaluation

A standardized dataloader plus eval protocol for scene flow datasets.

Currently supported datasets:

 - Argoverse 2
 - Waymo Open

## Installation

```
pip install bucketed-scene-flow-eval
```

## Setup

Follow our [Getting Started](docs/GETTING_STARTED.md) for setup instructions.

## Demo

We provide a demo script which shows off the various features of the API.

To run with Argoverse 2:

```
python scripts/demo.py --dataset Argoverse2SceneFlow --root_dir /efs/argoverse2/val/
```

To run with Waymo Open:

```
python scripts/demo.py --dataset WaymoOpenSceneFlow --root_dir /efs/waymo_open_processed_flow/validation/
```

## Evaluating AV2 flow submissions

To evaluate an AV2 Scene Flow challenge entry named `./submission_val.zip` against validation dataset masks `/efs/argoverse2/val_official_masks.zip`, run

```
python scripts/av2_eval.py /efs/argoverse2/val /efs/argoverse2/val_official_masks.zip ./submission_val.zip
```

## Documentation

See `docs/` for more documentation .