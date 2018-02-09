""" Initilize the datasets module
    New datasets can be added with python scripts under datasets/
"""
import torch
import torch.utils.data
import torch.utils.data.distributed
import importlib


def get_dataset(args):
    dataset = importlib.import_module('.'+args.dataset, package='datasets')
    train_dataset, val_dataset, valvideo_dataset = dataset.get(args)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(
            train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    valvideo_loader = torch.utils.data.DataLoader(
        valvideo_dataset, batch_size=25, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    return train_loader, val_loader, valvideo_loader
