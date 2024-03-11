#!/bin/bash
touch docker_history.txt
xhost +
docker run --gpus=all --rm -it \
 --shm-size=16gb \
 -v `pwd`:/bucketed_scene_flow_eval \
 -v /efs:/efs \
 -v /efs2:/efs2 \
 -v /bigdata:/bigdata \
 -v /tmp/.X11-unix:/tmp/.X11-unix \
 -v /tmp:/tmp \
 -v `pwd`/docker_history.txt:/root/.bash_history \
 -e DISPLAY=$DISPLAY \
 -h $HOSTNAME \
 --privileged \
 -p 6008:6008 \
 bucketed_scene_flow_eval:latest
