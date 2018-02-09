#!/usr/bin/env python
import sys
#sys.path.insert(0, '..')
sys.path.insert(0, '.')
from main import main

args = [
    '--name', __file__.split('/')[-1].split('.')[0],  # name is filename
    '--print-freq', '1',
    '--dataset', 'charadesflow',
    '--data','/scratch/gsigurds/Charades_v1_flow/',
    '--arch', 'vgg16flow',
    '--pretrained-weights', './vgg16flow_ucf101.pth',
    '--lr', '5e-3',
    '--lr-decay-rate','15',
    '--epochs','40',
    '--batch-size', '64',
    '--train-size', '0.2',
    '--val-size', '0.1',
    '--cache-dir', '/nfs.yoda/gsigurds/ai2/caches/',
    '--pretrained',
    #'--evaluate',
]
sys.argv.extend(args)
main()
