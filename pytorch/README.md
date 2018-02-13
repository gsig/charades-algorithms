## PyTorch Starter Code for Activity Classification and Localization on Charades

Contributor: Gunnar Atli Sigurdsson

Extension of this framework to the deep CRF model on Charades for *Asynchronous Temporal Fields for Action Recognition*: https://github.com/gsig/temporal-fields

* This code implements a Two-Stream network in PyTorch

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
yields a final classification accuracy of 20.6% mAP (Two-Stream) and on Charades (evalated with charades_v1_classify.m)


## Technical Overview:
 
The code is organized such that to train a two-stream network. Two independed network are trained: One RGB network and one Flow network.
This code parses the training data into pairs of an image (or flow), and a label for a single activity class. This forms a softmax training setup like a standard CNN. The network is a VGG-16 network. For RGB it is pretrained on Image-Net, and for Flow it is pretrained on UCF101. The pretrained networks can be downloaded with the scripts in this directory.
For testing. The network uses a batch size of 25, scores all images, and pools the output to make a classfication prediction or uses all 25 outputs for localization.

All outputs are stored in the cache-dir. This includes epoch*.txt which is the classification output, and localize*.txt which is the localization output (note the you need to specify that you want this in the options).
Those output files can be combined after training with the python scripts in this directory.
All output files can be scored with the official MATLAB evaluation script provided with the Charades dataset.

Requirements:
* Python 2.7
* PyTorch 


## Steps to train your own two-stream network on Charades:
 
1. Download the Charades Annotations (allenai.org/plato/charades/)
2. Download the Charades RGB and/or Flow frames (allenai.org/plato/charades/)
3. Duplicate and edit one of the experiment files under exp/ with appropriate parameters. For additional parameters, see opts.lua
4. Run an experiment by calling python exp/rgbnet.py where rgbnet.py is your experiment file
5. The checkpoints/logfiles/outputs are stored in your specified cache directory. 
6. Combine one RGB output file and one Flow output file with combine_rgb_flow.py to generate a submission file
7. Evaluate the submission file with the Charades_v1_classify.m or Charades_v1_localize.m evaluation scripts 
8. Build of the code, cite our papers, and say hi to us at CVPR.

Good luck!


## Pretrained networks:

While the RGB net can be trained in a day on a modern GPU, the flow net requires nontrivial IO and time to converge. For your convenience we provide RGB and Flow models already trained on Charades using exp/rgbnet.py and exp/flownet.py

https://www.dropbox.com/s/p457h2ifi6v1qdz/twostream_rgb.pth.tar?dl=1
https://www.dropbox.com/s/m1hkeiwjtndt26z/twostream_flow.pth?dl=1

* The rgb model was obtained after 7 epochs (epochSize=0.1)
* The rgb model has a classification accuracy of 18.6% mAP (evalated with charades_v1_classify.m)
* The flow model was converted directly from the Charades Torch codebase (../torch/)
* The flow model has a classification accuracy of 15.4% mAP (via charades_v1_classify.m)

Combining the predictions (submission files) of those models using combine_rgb_flow.py
yields a final classification accuracy of 20.6% mAP (evalated with charades_v1_classify.m)

To fine-tune those models, or run experiments, please see exp/rgbnet_test.py and exp/flownet_test.py


Additionally we include rgb-streams fine-tuned from resnet and inception pretrained on ImageNet:
* ResNet-152 (exp/rgbnet_resnet.py): 22.8% mAP (via charades_v1_classify.m)
* https://www.dropbox.com/s/iy9fmk0r1a3edoz/resnet_rgb.pth.tar?dl=1
* Inception_v3 (exp/rgbnet_inception.py): 22.7% mAP (via charades_v1_classify.m)
* https://www.dropbox.com/s/whxikophm7xqchb/inception_rgb.pth.tar?dl=1


Charades submission files are available for multiple baselines at https://github.com/gsig/temporal-fields
