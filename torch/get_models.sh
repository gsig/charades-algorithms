#!/bin/bash
# Script to download models pretrained on ImageNet and UCF101
# Those are used as the starting point for training on Charades

wget -O VGG_ILSVRC_16_layers_deploy.prototxt https://www.dropbox.com/s/iycrzeruaf75soc/VGG_ILSVRC_16_layers_deploy.prototxt?dl=1
wget -O VGG_UCF101_16_layers_deploy.prototxt https://www.dropbox.com/s/4ktxsdiiqm429j2/VGG_UCF101_16_layers_deploy.prototxt?dl=1
wget -O VGG_ILSVRC_16_layers.caffemodel https://www.dropbox.com/s/rwo3iim5z2w07aa/VGG_ILSVRC_16_layers.caffemodel?dl=1
wget -O VGG_UCF101_16_layers.caffemodel https://www.dropbox.com/s/d1n9emy0awzlwlr/VGG_UCF101_16_layers.caffemodel?dl=1
