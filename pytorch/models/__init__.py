"""
Initialize the model module
New models can be defined by adding scripts under models/
"""
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torchvision.models as tmodels
import importlib


def create_model(args):
    if args.arch in tmodels.__dict__:  # torchvision models
        if args.pretrained:
            print("=> using pre-trained model '{}'".format(args.arch))
            model = tmodels.__dict__[args.arch](pretrained=True)
            model = model.cuda()
        else:
            print("=> creating model '{}'".format(args.arch))
            model = tmodels.__dict__[args.arch]()
    else:  # defined as script in this directory
        model = importlib.import_module('.'+args.arch, package='models').model
        if not args.pretrained_weights == '':
            print('loading pretrained-weights from {}'.format(args.pretrained_weights))
            model.load_state_dict(torch.load(args.pretrained_weights))

    # replace last layer
    if hasattr(model, 'classifier'):
        newcls = list(model.classifier.children())
        newcls = newcls[:-1] + [nn.Linear(newcls[-1].in_features, args.nclass).cuda()]
        model.classifier = nn.Sequential(*newcls)
    elif hasattr(model, 'fc'):
        model.fc = nn.Linear(model.fc.in_features, args.nclass)
        if hasattr(model, 'AuxLogits'):
            model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, args.nclass)
    else:
        newcls = list(model.children())
        if hasattr(model, 'in_features'):
            in_features = model.in_features
        else:
            in_features = newcls[-1].in_features
        newcls = newcls[:-1] + [nn.Linear(in_features, args.nclass).cuda()]
        model = nn.Sequential(*newcls)

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        if hasattr(model, 'features'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    cudnn.benchmark = True
    return model, criterion, optimizer
