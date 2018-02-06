#!/usr/bin/env python
import sys
#sys.path.insert(0, '..')
sys.path.insert(0, '.')
from main import main

args = [
    '--name', __file__.split('/')[-1].split('.')[0],  # name is filename
    '--print-freq', '1',
    '--dataset', 'charadesrgb',
    '--arch', 'vgg16',
    '--lr', '1e-3',
    '--batch-size', '64',
    '--train-size', '0.1',
    '--val-size', '0.1',
    '--cache-dir', '/nfs.yoda/gsigurds/ai2/caches/',
    '--pretrained',
    '--resume', './twostream_rgb.pth.tar',
    '--evaluate',
]
sys.argv.extend(args)
main()
