## Charades Starter Code for Activity Recognition in Torch and PyTorch

Contributor: Gunnar Atli Sigurdsson

**New:** extension of this framework to the deep CRF model on Charades for *Asynchronous Temporal Fields for Action Recognition*: https://github.com/gsig/temporal-fields

* **New:** This code implements a Two-Stream network in PyTorch
* This code implements a Two-Stream network in Torch
* This code implements a Two-Stream+LSTM network in Torch

See [pytorch/](pytorch/), [torch/](torch/), for the code repositories.

The code replicates the 'Two-Stream Extended' and 'Two-Stream+LSTM' baselines found in:
```
@inproceedings{sigurdsson2017asynchronous,
author = {Gunnar A. Sigurdsson and Santosh Divvala and Ali Farhadi and Abhinav Gupta},
title = {Asynchronous Temporal Fields for Action Recognition},
booktitle={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
year={2017},
pdf = {http://arxiv.org/pdf/1612.06371.pdf},
code = {https://github.com/gsig/temporal-fields},
}
```
which is in turn based off "Two-stream convolutional networks for action recognition in videos" by Simonyan and Zisserman, and "Beyond Short Snippets: Deep Networks for Video Classification" by Joe Yue-Hei Ng el al.

Combining the predictions (submission files) of those models using combine_rgb_flow.py
yields a final classification accuracy of 18.9% mAP (Two-Stream) and 19.8% (LSTM) on Charades (evalated with charades_v1_classify.m)


## Technical Overview:
 
The code is organized such that to train a two-stream network. Two independed network are trained: One RGB network and one Flow network.
This code parses the training data into pairs of an image (or flow), and a label for a single activity class. This forms a softmax training setup like a standard CNN. The network is a VGG-16 network. For RGB it is pretrained on Image-Net, and for Flow it is pretrained on UCF101. The pretrained networks can be downloaded with the scripts in this directory.
For testing. The network uses a batch size of 25, scores all images, and pools the output to make a classfication prediction or uses all 25 outputs for localization.

