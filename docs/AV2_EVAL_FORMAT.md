# AV2 Eval Format

This repo provides the stand alone evaluation infrastructure for the [Argoverse 2 2024 Scene Flow Challenge](https://argoverse.org/sceneflow). The script used for evaluation is `scripts/evals/av2_eval.py`.

This script makes several important assumptions:

 - The input zip folder is structured like the supervised AV2 scene flow labels, but with only every 5th frame provided (this reduces submission size).
 - Every feather file is in the same format and same frame as the supervised labels, but without the `classes` column.
