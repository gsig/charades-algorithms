## Charades Starter Code for Activity Classification and Localization

Contributor: Gunnar Atli Sigurdsson

**New:** extension of this framework to the deep CRF model on Charades for *Asynchronous Temporal Fields for Action Recognition*: https://github.com/gsig/temporal-fields

* This code implements a Two-Stream network in Torch
* This code implements a Two-Stream+LSTM network in Torch
* This code is built of the Res-Net Torch source code: github.com/facebook/fb.resnet.torch
* This code awkwardly hacks said code to work as Two-Stream/LSTM
* Some functionality from original code may work (optnet)
* Some functionality from original code may not work (resnet)

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

All outputs are stored in the cacheDir under checkpoints/. This includes epoch*.txt which is the classification output, and localize*.txt which is the localization output (note the you need to specify that you want this in the options).
Those output files can be combined after training with the python scripts in this directory.
All output files can be scored with the official MATLAB evaluation script provided with the Charades dataset.

Requirements:
* csvigo: luarocks install csvigo
* loadcaffe: luarocks install loadcaffe
* optnet: luarocks install optnet 
(The flow net requires optnet to converge with the current default settings for the parameters)

Optional requirements:
* Facebook Lua Libraries, for speedups and fb.debugger, a great debugger
Please refer to the original res-net codebase for more information.


## Steps to train your own two-stream network on Charades:
 
1. Download the Charades Annotations (allenai.org/plato/charades/)
2. Download the Charades RGB and/or Flow frames (allenai.org/plato/charades/)
3. Download the Imagenet/UCF101 Pre-trained Image and Flow models using ./get_models.sh
4. Duplicate and edit one of the experiment files under exp/ with appropriate parameters. For additional parameters, see opts.lua
5. Run an experiment by calling dofile 'exp/rgbnet.lua' where rgbnet.lua is your experiment file
6. The checkpoints/logfiles/outputs are stored in your specified cache directory. 
7. Combine one RGB output file and one Flow output file with combine_rgb_flow.py to generate a submission file
8. Evaluate the submission file with the Charades_v1_classify.m or Charades_v1_localize.m evaluation scripts 
9. Build of the code, cite our papers, and say hi to us at CVPR.

Good luck!


## Pretrained networks:

While the RGB net can be trained in a day on a modern GPU, the flow net requires nontrivial IO and time to converge. For your convenience we provide RGB and Flow models already trained on Charades using exp/rgbnet.lua and exp/flownet.lua

https://www.dropbox.com/s/o7afkhw52rqr48g/twostream_flow.t7?dl=1
https://www.dropbox.com/s/bo9rv32zaxojsmz/twostream_rgb.t7?dl=1

* The flow model was obtained after 31 epochs (epochSize=0.2)
* The flow model has a classification accuracy of 15.4% mAP (evalated with charades_v1_classify.m)
* The rgb model was obtained after 6 epochs (epochSize=0.1)
* The rgb model has a classification accuracy of 15.6% mAP (evalated with charades_v1_classify.m)

Combining the predictions (submission files) of those models using combine_rgb_flow.py
yields a final classification accuracy of 18.9% mAP (evalated with charades_v1_classify.m)

To fine-tune those models, or run experiments, please see exp/rgbnet_resume.lua, exp/rgbnet_test.lua, exp/flownet_resume.lua, and exp/flownet_test.lua

Charades submission files are available for multiple baselines at https://github.com/gsig/temporal-fields

## Two-Stream+LSTM details

We also provide pre-trained LSTM models using exp/lstmrgbnet.lua and exp/lstmflownet.lua, please see get_alreadytrained_lstm.sh for details.

This baseline fine-tunes the previous Two-Stream models with a LSTM on top of fc7. It uses a special loader for Charades (charadessync), that feeds in a full video for each batch, to train an LSTM. To accomodate the softmax loss, (frame,label) pairs are randomly sampled for the training set. exp/lstmrgnet.lua, models/vgg16lstm.lua, and datasets/charadessync-gen.lua contain more details.
