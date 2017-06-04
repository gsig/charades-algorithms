#!/usr/bin/env python
#
# Script for combining the submission files for the RGB and Flow networks
#
# Contributor: Gunnar Atli Sigurdsson

import numpy as np
import sys
import pdb
from itertools import groupby

rgbfile = sys.argv[1]
flowfile = sys.argv[2]
w = [0.5,0.5]
nclasses = 157

def loadfile(path):
    with open(path) as f:
        lines = [x.strip().split(' ') for x in f.readlines()]
    localization = len(lines[0]) == nclasses+2
    if localization:
        data = [(x[0]+' '+x[1],np.array([float(y) for y in x[2:]])) for x in lines]
    else:
        data = [(x[0],np.array([float(y) for y in x[1:]])) for x in lines]
    return data

rgb = loadfile(rgbfile)
flow = loadfile(flowfile)

rgbdict = dict(rgb)
flowdict = dict(flow)

keys = list(set(rgbdict.keys()+flowdict.keys()))
w = [x/sum(w) for x in w]

def normme(x):
    x = x-np.mean(x)
    x = x/(0.00001+np.std(x))
    return x

N = 157
def lookup(d,key):
    if key in d:
        return d[key]
    else:
        sys.stderr.write('error ' + key + '\n')
        return np.zeros((nclasses,))

for id0 in keys:
    r = lookup(rgbdict,id0)
    f = lookup(flowdict,id0)
    out = r*w[0]+f*w[1] #unnormalized combination
    #out = normme(r)*w[0]+normme(f)*w[1] #normalize first
    #out = np.exp(np.log(r)*w[0]+np.log(f)*w[1]) #weighted geometric mean
    out = [str(x) for x in out]
    print('{} {}'.format(id0,' '.join(out)))
