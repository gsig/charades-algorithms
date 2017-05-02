#!/bin/bash
# Script to download models pretrained on ImageNet and UCF101
# Those are used as the starting point for training on Charades

wget https://dl.dropboxusercontent.com/u/10728218/models/VGG_ILSVRC_16_layers_deploy.prototxt
wget https://dl.dropboxusercontent.com/u/10728218/models/VGG_UCF101_16_layers_deploy.prototxt
wget https://dl.dropboxusercontent.com/u/10728218/models/VGG_ILSVRC_16_layers.caffemodel
wget https://dl.dropboxusercontent.com/u/10728218/models/VGG_UCF101_16_layers.caffemodel
