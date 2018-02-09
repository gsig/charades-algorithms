#!/usr/bin/env python

"""Charades activity recognition baseline code
   Can be run directly or throught config scripts under exp/

   Gunnar Sigurdsson, 2018
""" 
import torch
import numpy as np
import random
import train
from models import create_model
from datasets import get_dataset
import checkpoints
from opts import parse
from utils import tee


def seed(manualseed):
    random.seed(manualseed)
    np.random.seed(manualseed)
    torch.manual_seed(manualseed)
    torch.cuda.manual_seed(manualseed)


best_mAP = 0
def main():
    global opt, best_mAP
    opt = parse()
    tee.Tee(opt.cache+'/log.txt')
    print(vars(opt))
    seed(opt.manual_seed)

    model, criterion, optimizer = create_model(opt)
    if opt.resume: best_mAP = checkpoints.load(opt, model, optimizer)
    print(model)
    trainer = train.Trainer()
    train_loader, val_loader, valvideo_loader = get_dataset(opt)

    if opt.evaluate:
        trainer.validate(val_loader, model, criterion, -1, opt)
        trainer.validate_video(valvideo_loader, model, -1, opt)
        return

    for epoch in range(opt.start_epoch, opt.epochs):
        if opt.distributed:
            trainer.train_sampler.set_epoch(epoch)
        top1,top5 = trainer.train(train_loader, model, criterion, optimizer, epoch, opt)
        top1val,top5val = trainer.validate(val_loader, model, criterion, epoch, opt)
        mAP = trainer.validate_video(valvideo_loader, model, epoch, opt)
        is_best = mAP > best_mAP
        best_mAP = max(mAP, best_mAP)
        scores = {'top1train':top1,'top5train':top5,'top1val':top1val,'top5val':top5val,'mAP':mAP}
        checkpoints.save(epoch, opt, model, optimizer, is_best, scores)


if __name__ == '__main__':
    main()
