""" Defines functions used for checkpointing models and storing model scores
"""
import os
import torch
import shutil


def load(args, model, optimizer):
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_mAP = checkpoint['best_mAP']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            return best_mAP
        else:
            raise ValueError("no checkpoint found at '{}'".format(args.resume))
    return 0

def score_file(scores,filename):
    with open(filename, 'w') as f:
        for key,val in sorted(scores.items()):
            f.write('{} {}\n'.format(key,val))

def save(epoch, args, model, optimizer, is_best, scores):
    state = {
        'epoch': epoch + 1,
        'arch': args.arch,
        'state_dict': model.state_dict(),
        'mAP': scores['mAP'],
        'optimizer': optimizer.state_dict(),
    }
    filename = "{}/model.pth.tar".format(args.cache)
    score_file(scores, "{}/model_{:03d}.txt".format(args.cache,epoch+1))
    torch.save(state, filename)
    if is_best:
        bestname = "{}/model_best.pth.tar".format(args.cache)
        score_file(scores, "{}/model_best.txt".format(args.cache,epoch+1))
        shutil.copyfile(filename, bestname)
