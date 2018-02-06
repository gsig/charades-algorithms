#!/bin/bash
# Script to download pretrained pytorch models on Charades
# Approximately equivalent to models obtained by running exp/rgbnet.py
#
# The rgb model was obtained after 7 epochs (epoch-size 0.1)
# The rgb model has a classification accuracy of 15.9% mAP (via charades_v1_classify.m)

wget -O twostream_rgb.pth.tar https://www.dropbox.com/s/p457h2ifi6v1qdz/twostream_rgb.pth.tar?dl=1
