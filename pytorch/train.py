""" Defines the Trainer class which handles train/validation/validation_video
"""
import time
import torch
import itertools
import numpy as np
from utils import map


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(startlr, decay_rate, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = startlr * (0.1 ** (epoch // decay_rate))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def submission_file(ids, outputs, filename):
    """ write list of ids and outputs to filename"""
    with open(filename, 'w') as f:
        for vid, output in zip(ids, outputs):
            scores = ['{:g}'.format(x)
                      for x in output]
            f.write('{} {}\n'.format(vid, ' '.join(scores)))


class Trainer():
    def train(self, loader, model, criterion, optimizer, epoch, args):
        adjust_learning_rate(args.lr, args.lr_decay_rate, optimizer, epoch)
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to train mode
        model.train()
        optimizer.zero_grad()

        def part(x): return itertools.islice(x, int(len(x)*args.train_size))
        end = time.time()
        for i, (input, target, meta) in enumerate(part(loader)):
            data_time.update(time.time() - end)

            target = target.long().cuda(async=True)
            input_var = torch.autograd.Variable(input.cuda())
            target_var = torch.autograd.Variable(target)
            output = model(input_var)
            loss = None
            # for nets that have multiple outputs such as inception
            if isinstance(output, tuple):
                loss = sum((criterion(o,target_var) for o in output))
                output = output[0]
            else:
                loss = criterion(output, target_var)
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data[0], input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            loss.backward()
            if i % args.accum_grad == args.accum_grad-1:
                print('updating parameters')
                optimizer.step()
                optimizer.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}({3})]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                          epoch, i, int(
                              len(loader)*args.train_size), len(loader),
                          batch_time=batch_time, data_time=data_time, loss=losses, top1=top1, top5=top5))
        return top1.avg,top5.avg

    def validate(self, loader, model, criterion, epoch, args):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        model.eval()

        def part(x): return itertools.islice(x, int(len(x)*args.val_size))
        end = time.time()
        for i, (input, target, meta) in enumerate(part(loader)):
            target = target.long().cuda(async=True)
            input_var = torch.autograd.Variable(input.cuda(), volatile=True)
            target_var = torch.autograd.Variable(target, volatile=True)
            output = model(input_var)
            loss = criterion(output, target_var)

            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data[0], input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1} ({2})]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                          i, int(len(loader)*args.val_size), len(loader),
                          batch_time=batch_time, loss=losses,
                          top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

        return top1.avg,top5.avg

    def validate_video(self, loader, model, epoch, args):
        """ Run video-level validation on the Charades test set"""
        batch_time = AverageMeter()
        outputs = []
        gts = []
        ids = []

        # switch to evaluate mode
        model.eval()

        end = time.time()
        for i, (input, target, meta) in enumerate(loader):
            target = target.long().cuda(async=True)
            assert target[0,:].eq(target[1,:]).all(), "val_video not synced"
            input_var = torch.autograd.Variable(input.cuda(), volatile=True)
            output = model(input_var)
            output = torch.nn.Softmax(dim=1)(output)

            # store predictions
            output_video = output.mean(dim=0)
            outputs.append(output_video.data.cpu().numpy())
            gts.append(target[0,:])
            ids.append(meta['id'][0])
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test2: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                          i, len(loader), batch_time=batch_time))
        #mAP, _, ap = map.map(np.vstack(outputs), np.vstack(gts))
        mAP, _, ap = map.charades_map(np.vstack(outputs), np.vstack(gts))
        print(ap)
        print(' * mAP {:.3f}'.format(mAP))
        submission_file(
            ids, outputs, '{}/epoch_{:03d}.txt'.format(args.cache, epoch+1))
        return mAP
