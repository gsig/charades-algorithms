#!/bin/bash
# Script to download pretrained pytorch models on Charades
# Approximately equivalent to models obtained by running exp/rgbnet.py
#
# The rgb model was obtained after 7 epochs (epoch-size 0.1)
# The rgb model has a classification accuracy of 18.6% mAP (via charades_v1_classify.m)
#     Notice that this is an improvement over the Torch RGB model
# The flow model was converted directly from the Charades Torch codebase (../torch/)
# The flow model has a classification accuracy of 15.4% mAP (via charades_v1_classify.m)
#
# vgg16flow_ucf101.pth is a converted model from Torch that was pretrained on UCF101
# and is used as an initialization for the flow model
#
# Combining the predictions (submission files) of those models using combine_rgb_flow.py
# yields a final classification accuracy of 20.6% mAP (via charades_v1_classify.m)
#
# Additionally we include rgb-streams fine-tuned from resnet and inception pretrained on ImageNet
# ResNet-152 (exp/rgbnet_resnet.py): 22.8% mAP (via charades_v1_classify.m)
# Inception_v3 (exp/rgbnet_inception.py): 22.7% mAP (via charades_v1_classify.m)

wget -O twostream_rgb.pth.tar https://www.dropbox.com/s/p457h2ifi6v1qdz/twostream_rgb.pth.tar?dl=1
wget -O twostream_flow.pth https://www.dropbox.com/s/m1hkeiwjtndt26z/twostream_flow.pth?dl=1
wget -O vgg16flow_ucf101.pth https://www.dropbox.com/s/qlr5aty2jz4dq5o/vgg16flow_ucf101.pth?dl=1
wget -O resnet_rgb.pth.tar https://www.dropbox.com/s/iy9fmk0r1a3edoz/resnet_rgb.pth.tar?dl=1
wget -O inception_rgb.pth.tar https://www.dropbox.com/s/whxikophm7xqchb/inception_rgb.pth.tar?dl=1
