#!/bin/bash

# Setup fake environment


# Prepare /tmp/argoverse2_tiny/val
rm -rf /tmp/argoverse2_tiny
echo "Downloading argoverse2_tiny"
wget -q https://github.com/kylevedder/BucketedSceneFlowEval/files/14576619/argoverse2_tiny.zip -O /tmp/argoverse2_tiny.zip
unzip -q /tmp/argoverse2_tiny.zip -d /tmp/

# For testing the raw data loader only (no flow) mode of the argoverse2 dataset, we symlink a "test" split to the val split
ln -s /tmp/argoverse2_tiny/val /tmp/argoverse2_tiny/test

# Prepare /tmp/waymo_open_processed_flow_tiny
rm -rf /tmp/waymo_open_processed_flow_tiny
echo "Downloading waymo_open_processed_flow_tiny"
wget -q https://github.com/kylevedder/BucketedSceneFlowEval/files/13924555/waymo_open_processed_flow_tiny.zip -O /tmp/waymo_open_processed_flow_tiny.zip
unzip -q /tmp/waymo_open_processed_flow_tiny.zip -d /tmp/


# Prepare /tmp/nuscenes v1.0-mini
rm -rf /tmp/nuscenes
mkdir -p /tmp/nuscenes
echo "Downloading nuscenes v1.0-mini"
wget -q https://www.nuscenes.org/data/v1.0-mini.tgz -O /tmp/nuscenes/nuscenes_v1.0-mini.tgz
tar -xzf /tmp/nuscenes/nuscenes_v1.0-mini.tgz -C /tmp/nuscenes

echo "Running integration tests"
pytest tests/integration_tests.py tests/eval/bucketed_epe.py tests/eval/threeway_epe.py tests/nuscenes/nuscenes_tests.py tests/argoverse2/av2_tests.py
