#!/bin/bash

# Setup fake environment


# Make /tmp/argoverse2/val

wget https://github.com/kylevedder/zeroflow/files/13059582/argoverse2_tiny.zip -O /tmp/argoverse2_tiny.zip
rm -rf /tmp/argoverse2_tiny
unzip /tmp/argoverse2_tiny.zip -d /tmp/
python tests/integration_tests.py
