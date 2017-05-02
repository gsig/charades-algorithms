#!/bin/bash
# Script to download pretrained models on Charades
# Approximately equivalent to models obtained by running exp/rgbnet.lua and exp/flownet.lua
#
# The flow model was obtained after 31 epochs (epochSize=0.2)
# The flow model has a classification accuracy of 15.4% mAP (via charades_v1_classify.m)
# The rgb model was obtained after 6 epochs (epochSize=0.1)
# The rgb model has a classification accuracy of 15.6% mAP (via charades_v1_classify.m)
#
# Combining the predictions (submission files) of those models using combine_rgb_flow.py
# yields a final classification accuracy of 18.9% mAP (via charades_v1_classify.m)

wget https://dl.dropboxusercontent.com/u/10728218/models/twostream_flow.t7
wget https://dl.dropboxusercontent.com/u/10728218/models/twostream_rgb.t7
