#!/bin/bash

# Setup fake environment


# Prepare /tmp/argoverse2_tiny/val
rm -rf /tmp/argoverse2_tiny
wget https://github.com/kylevedder/BucketedSceneFlowEval/files/13881746/argoverse2_tiny.zip -O /tmp/argoverse2_tiny.zip
unzip -q /tmp/argoverse2_tiny.zip -d /tmp/

# Prepare /tmp/waymo_open_processed_flow_tiny
rm -rf /tmp/waymo_open_processed_flow_tiny
wget https://github.com/kylevedder/BucketedSceneFlowEval/files/13924555/waymo_open_processed_flow_tiny.zip -O /tmp/waymo_open_processed_flow_tiny.zip
unzip -q /tmp/waymo_open_processed_flow_tiny.zip -d /tmp/

pytest tests/integration_tests.py
