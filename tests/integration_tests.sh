#!/bin/bash

# Setup fake environment


# Prepare /tmp/argoverse2/val
rm -rf /tmp/argoverse2_tiny
wget https://github.com/kylevedder/BucketedSceneFlowEval/files/13881746/argoverse2_tiny.zip -O /tmp/argoverse2_tiny.zip
unzip -q /tmp/argoverse2_tiny.zip -d /tmp/
python tests/integration_tests.py
