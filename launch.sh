#!/bin/bash
touch docker_history.txt
xhost +
docker run --gpus=all --rm -it \
 --shm-size=16gb \
 -v `pwd`:/bucketed_scene_flow_eval \
 -v /efs:/efs \
 -v /tmp/.X11-unix:/tmp/.X11-unix \
 -v /tmp:/tmp \
 -v `pwd`/docker_history.txt:/root/.bash_history \
 -e DISPLAY=$DISPLAY \
 -h $HOSTNAME \
 --privileged \
 bucketed_scene_flow_eval:latest
