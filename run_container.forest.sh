#!/usr/bin/env bash

docker run \
	--gpus all \
	--mount type=bind,source="$(pwd)",target=/layerjot/pytorch-metric-learning \
	--mount type=bind,source="/home/data",target=/data \
	--mount type=bind,source="/home/results",target=/results \
	--rm --shm-size 2G --network=host -it pymetlearn:latest

