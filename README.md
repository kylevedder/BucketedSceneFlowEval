# Bucketed Scene Flow Evaluation

This repo provides:
 - A speed and class aware evaluation protocol called _Bucketed Scene Flow Evaluation_
 - A standardized interface for working with Scene Flow datasets.

Currently supported datasets:

 - Argoverse 2 (Human Labeled and NSFP Pseudolabeled)
 - Waymo Open

## The _Bucketed Scene Flow Evaluation_ protocol

The _Bucketed Scene Flow Evaluation_ protocol is designed to quantitatively measure the failure of state-of-the-art scene flow methods to properly capture motion on smaller objects. In the Autononous Vehicle domain, SotA methods almost universially fail on important objects like Pedestrians and Bicyclists. As part of organizing the community around addressing these issues, the [Argoverse 2 2024 Scene Flow Challenge](https://www.argoverse.org/sceneflow) uses this protocol to evaluate submissions. More details about the protocol can be found [in the challenge blogpost](https://www.argoverse.org/sceneflow).

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
