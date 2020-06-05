#!/bin/bash
REPO=image-common.com:9487
CONTAINER=pytorch_dev
TAG=9487
DOCKER_IMAGE=$REPO/$CONTAINER:$TAG

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BUILDROOT=$DIR/..

# run
cmd="nvidia-docker run -it --rm \
    -v $BUILDROOT:/workspace \
    $DOCKER_IMAGE bash
"

echo $cmd
eval $cmd
