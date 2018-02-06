""" 
Initialize the model module
New models can be defined by adding scripts under models/ 
"""
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torchvision.models as models


def create_model(args):
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
        model = model.cuda()
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

        if args.distributed:
            dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                    world_size=args.world_size)

        if not args.distributed:
            if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
                model.features = torch.nn.DataParallel(model.features)
                model.cuda()
            else:
                model = torch.nn.DataParallel(model).cuda()
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)

    # replace last layer
    newcls = list(model.classifier.children())[:-1]
    newcls += [nn.Linear(4096, args.nclass).cuda()]
    model.classifier = nn.Sequential(*newcls)

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    cudnn.benchmark = True
    return model, criterion, optimizer
