# Program to plot the trace of hessian
#!/usr/bin/env python3

import os
import sys
import time
import torch
import logging
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import os.path as osp
import torch.nn as nn
import torch.optim as optim
import torch.autograd as AG
import torch.utils.data as data
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms
import torchvision.datasets as datasets

sys.path.append(osp.dirname(os.getcwd()))
from models.cifar import Network
from utils import Config
from utils import get_mean_std
from run_models import model_choices


def parse_args():

    parser = argparse.ArgumentParser(description='train')

    dataset_choices = ['CIFAR10', 'CIFAR100', 'MNIST', 'FashionMNIST', 'STL10', 'SVHN']
    optimizer_choices = ['sgd', 'adam']

    default_learning_rate = 1e-4
    default_l2 = 0.0
    default_num_epochs = 100
    default_optimizer = 'sgd'
    default_dataset = dataset_choices[0]
    default_batch_size = 256
    default_workers = 4
    default_model = model_choices[0]
    default_milestone = [1000000]
    default_step_gamma = 0.1
    default_dataset_root = osp.join(osp.dirname(os.getcwd()) ,'datasets')
    default_log_dir = 'log'
    default_ckpt_dir = 'ckpt'
    default_images_dir = 'images'
    default_seed = 0
    default_sp = 1000000
    default_sf = 1e-3
    default_freq = 50
    default_reruns = 4
    default_hi = 100

    parser.add_argument('-lr',
                        '--learning_rate',
                        type=float,
                        default=default_learning_rate,
                        help='learning rate, default={}'.format(default_learning_rate)
                        )

    parser.add_argument('-l2',
                        '--weight_decay',
                        type=float,
                        default=default_l2,
                        help='l2 penalty, default={}'.format(default_l2)
                        )

    parser.add_argument('-n',
                        '--num_epochs',
                        type=int,
                        default=default_num_epochs,
                        help='number of training epochs, default={}'.format(default_num_epochs)
                        )

    parser.add_argument('-o',
                        '--optimizer',
                        type=str,
                        default=default_optimizer,
                        choices=['sgd', 'adam'],
                        help='optimizer, default={}'.format(default_optimizer)
                        )

    parser.add_argument('-d',
                        '--dataset',
                        type=str,
                        choices=dataset_choices,
                        default=default_dataset,
                        help='type of dataset, default={}'.format(default_dataset)
                        )

    parser.add_argument('-pdb',
                        '--with_pdb',
                        action='store_true',
                        help='run with python debugger'
                        )

    parser.add_argument('-b',
                        '--batch_size',
                        type=int,
                        default=default_batch_size,
                        help='batch size for training, default={}'.format(default_batch_size)
                        )

    parser.add_argument('-j',
                        '--workers',
                        type=int,
                        default=default_workers,
                        help='number of wrokers for dataloader, default={}'.format(default_workers)
                        )

    parser.add_argument('--dataset_root',
                        type=str,
                        default=default_dataset_root,
                        help='directory for dataset, default={}'.format(default_dataset_root)
                        )

    parser.add_argument('--log_dir',
                        type=str,
                        default=default_log_dir,
                        help='directory for logs, default={}'.format(default_log_dir)
                        )

    parser.add_argument('--ckpt_dir',
                        type=str,
                        default=default_ckpt_dir,
                        help='directory to store checkpoints, '
                             'default={}'.format(default_ckpt_dir)
                        )

    parser.add_argument('--images_dir',
                        type=str,
                        default=default_images_dir,
                        help='directory to store images'
                             ', default={}'.format(default_images_dir)
                        )

    parser.add_argument('-m',
                        '--model',
                        type=str,
                        default=default_model,
                        choices=model_choices,
                        help='model type, default={}'.format(default_model)
                        )

    parser.add_argument('--cuda',
                        type=int,
                        help='use cuda, if use, then give gpu number'
                        )

    parser.add_argument('--loss',
                        type=str,
                        default='ce',
                        choices=['ce'],
                        help='loss name, default=ce'
                        )

    parser.add_argument('-r',
                        '--run',
                        type=str,
                        help='run directory prefix'
                        )

    parser.add_argument('--save_freq',
                        type=int,
                        default=default_freq,
                        help='save epoch weights with these freq, default=None'
                        )

    # parser.add_argument('--milestones',
    #                     type=int,
    #                     nargs='+',
    #                     default=default_milestone,
    #                     help='milestones for multistep-lr scheduler, '
    #                     'default={}'.format(default_milestone)
    #                     )

    # parser.add_argument('--step_gamma',
    #                     type=float,
    #                     default=default_step_gamma,
    #                     help='gamma for step-lr scheduler'
    #                     ', default={}'.format(default_step_gamma)
    #                     )

    # parser.add_argument('--augment',
    #                     action='store_true',
    #                     help='augment data with random-flip and random crop'
    #                     )

    # parser.add_argument('--resume',
    #                     type=str,
    #                     help='path to *.pth to resume training'
    #                     )

    parser.add_argument('--seed',
                        type=int,
                        default=default_seed,
                        help='seed for randomness, default={}'.format(default_seed)
                        )

    parser.add_argument('--sp',
                        type=int,
                        default=default_sp,
                        help='scale point is epoch before which hessian has to be scaled, default={}'.format(default_sp)
                        )

    parser.add_argument('--sf',
                        type=float,
                        default=default_sf,
                        help='scale factor is the scaling constant of hessian, default={}'.format(default_sf)
                        )

    # parser.add_argument('--abs',
    #                     action='store_true',
    #                     help='peanlize sum of absolute(H_{ii})'
    #                     )

    parser.add_argument('-f', 
                        '--freq',
                        type=int,
                        default=default_freq,
                        help='penalization frequency, default={}'.format(default_freq)
                        )

    parser.add_argument('-reruns',
                        type=int,
                        default=default_reruns,
                        help='number of reruns, default={}'.format(default_reruns)
                        )

    parser.add_argument('--hi', 
                        type=int,
                        default=default_hi,
                        help='number of iterations for hutchingson, default={}'.format(default_hi)
                        )

    return parser.parse_args()

def get_trace_loss(model, batch, truth, criterion, hi):

    niters = hi
    V = list()
    for _ in range(niters):
        V_i = [torch.randint_like(p, high=2, device=device) for p in model.parameters()]
        for V_ij in V_i:
            V_ij[V_ij == 0] = -1
        V.append(V_i)
    trace = list()
    for V_i in V:
        y = model(batch)
        loss = criterion(y, truth)
        grad = AG.grad(loss, model.parameters(), create_graph=True)
        Hv = AG.grad(grad, model.parameters(), V_i, create_graph=True)
        this_trace = 0.0
        for Hv_, V_i_ in zip(Hv, V_i):
            this_trace = this_trace + torch.sum(Hv_ * V_i_)
        trace.append(this_trace)
    r = sum(trace) / niters
    return r.detach().cpu().numpy()

def evaluate_model(model, criterion, dataloader, device, dataset_size, args):

    model.eval()
    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        for batch, truth in dataloader:
            batch = batch.to(device)
            truth = truth.to(device)

            output = model(batch)
            _, preds = torch.max(output, 1)
            running_corrects += torch.sum(preds == truth)
                
            loss = criterion(output, truth)

            running_loss += loss.item() * batch.size(0)
    return {'loss': running_loss / dataset_size, 'acc': running_corrects.double() / dataset_size}


def train(model,
          optimizer,
        #   scheduler,
          dataloaders,
          criterion,
          device,
          sp,
          sf,
          train_type,
          num_epochs=100,
          args=None,
          dataset_sizes={'train': 5e4, 'test': 1e4},
          images_dir=None,
          ckpt_dir=None
          ):

    logger = logging.getLogger('train')
    loss_list = {'train': list(), 'test': list()}
    acc_list = {'train': list(), 'test': list()}
    trace_list = []
    assert images_dir is not None
    assert ckpt_dir is not None

    model.train()
    for epoch in range(num_epochs):
        logger.info('epoch: %d' % epoch)
        with torch.enable_grad():
            for step, (batch, truth) in enumerate(dataloaders['train']):
                batch = batch.to(device)
                truth = truth.to(device)
                optimizer.zero_grad()
                output = model(batch)
                loss = criterion(output, truth)
                loss.backward()
                optimizer.step()
        trace = get_trace_loss(model, batch, truth, criterion, args.hi)
        print(trace)
        trace_list.append(trace)
        # scheduler.step()
        for phase in ['train', 'test']:

            stats = evaluate_model(model, criterion, dataloaders[phase], device, dataset_sizes[phase], args)

            loss_list[phase].append(stats['loss'])
            acc_list[phase].append(stats['acc'])

            logger.info('{}:'.format(phase))
            logger.info('\tloss:{}'.format(stats['loss']))
            logger.info('\tacc :{}'.format(stats['acc']))

    return acc_list, loss_list, trace_list


if __name__ == '__main__':

    args = parse_args()
    if args.with_pdb:
        import pdb
        pdb.set_trace()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    sns.set_style('darkgrid')
    if args.cuda is None:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:%d' % args.cuda)
    log_dir = osp.join(args.run, 'logs')
    ckpt_dir = osp.join(args.run, 'ckpt')
    images_dir = osp.join(args.run, 'images')

    if not osp.exists(args.run):
        os.makedirs(args.run)
    if not osp.exists(log_dir):
        os.makedirs(log_dir)
    if not osp.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    if not osp.exists(images_dir):
        os.makedirs(images_dir)

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logging_file = osp.join(log_dir, 'train.log')
    logger = logging.getLogger('train')
    with open(logging_file, 'w+') as f:
        pass
    logger_file_handler = logging.FileHandler(logging_file)
    logger.addHandler(logger_file_handler)
    logger.info('Arguments: {}'.format(args))

    mean, std = get_mean_std(args.dataset)

    if args.dataset in ['MNIST', 'FashionMNIST']:
        input_ch = 1
        padded_im_size = 32
        num_classes = 10
        im_size = 28
        epc_seed = 0
        config = Config(input_ch=input_ch, 
                    padded_im_size=padded_im_size, 
                    num_classes=num_classes,
                    im_size=im_size
                    )
        dataset_sizes = {'train': 6e4, 'test': 1e4}
    elif args.dataset in ['CIFAR10', 'CIFAR100']:
        input_ch = 3
        padded_im_size = 32
        if args.dataset == 'CIFAR10':
            num_classes = 10
        elif args.dataset == 'CIFAR100':
            num_classes = 100
        else:
            raise Exception('Should not have reached here')
        im_size = 32
        epc_seed = 0
        config = Config(input_ch=input_ch, 
                    padded_im_size=padded_im_size,
                    num_classes=num_classes,
                    im_size=im_size
                    )
        dataset_sizes = {'train': 5e4, 'test': 1e4}
    elif args.dataset in ['STL10']:
        input_ch = 3
        padded_im_size = 102
        num_classes = 10
        im_size = 96
        epc_seed = 0
        config = Config(input_ch=input_ch,
                padded_im_size=padded_im_size,
                num_classes=num_classes,
                im_size=im_size
                )
        dataset_sizes = {'train': 5000, 'test': 8000}
    elif args.dataset in ['SVHN']:
        input_ch = 3
        padded_im_size = 32
        num_classes = 11
        im_size = 32
        epc_seed = 0
        config = Config(input_ch=input_ch,
                    padded_im_size=padded_im_size,
                    num_classes=num_classes,
                    im_size=im_size
                    )
        dataset_sizes = {'train': 73257, 'test': 26032}
    else:    
        raise Exception('Should not have reached here')
    pad = (config.padded_im_size - config.im_size) // 2

    all_acc  = {'train': {'ord': list(), 'ht': list()}, 'test': {'ord': list(), 'ht': list()}}
    all_loss = {'train': {'ord': list(), 'ht': list()}, 'test': {'ord': list(), 'ht': list()}}

    for rerun in range(args.reruns):
        for train_type in ['ord']:
            np.random.seed(args.seed + rerun)
            torch.manual_seed(args.seed + rerun)
            # if args.augment:
            #     train_transform = transforms.Compose([
            #         transforms.Pad(pad),
            #         transforms.ColorJitter(),
            #         transforms.RandomCrop(config.padded_im_size, padding=pad),
            #         transforms.RandomRotation(30),
            #         transforms.RandomHorizontalFlip(),
            #         transforms.ToTensor(),
            #         transforms.Normalize(mean, std),
            #     ])
            # else:
            train_transform = transforms.Compose([
                transforms.Pad(pad),
                transforms.ToTensor(), 
                transforms.Normalize(mean, std)])
            test_transform = transforms.Compose([
                    transforms.Pad(pad),
                    transforms.ToTensor(), 
                    transforms.Normalize(mean, std)]
                )


            if args.dataset == 'MNIST':
                train_data = datasets.MNIST(osp.join(args.dataset_root, 'MNIST'), train=True, transform=train_transform, download=False)
                test_data = datasets.MNIST(osp.join(args.dataset_root, 'MNIST'), train=False, transform=test_transform, download=False)
            elif args.dataset == 'FashionMNIST':
                train_data = datasets.FashionMNIST(osp.join(args.dataset_root, 'FashionMNIST'), train=True, transform=train_transform, download=False)
                test_data = datasets.FashionMNIST(osp.join(args.dataset_root, 'FashionMNIST'), train=False, transform=test_transform, download=False)
            elif args.dataset == 'CIFAR10':
                train_data = datasets.CIFAR10(osp.join(args.dataset_root, 'CIFAR10'), train=True, transform=train_transform, download=False)
                test_data = datasets.CIFAR10(osp.join(args.dataset_root, 'CIFAR10'), train=False, transform=test_transform, download=False)
            elif args.dataset == 'CIFAR100':
                train_data = datasets.CIFAR100(osp.join(args.dataset_root, 'CIFAR100'), train=True, transform=train_transform, download=False)
                test_data = datasets.CIFAR100(osp.join(args.dataset_root, 'CIFAR100'), train=False, transform=test_transform, download=False)
            elif args.dataset == 'STL10':
                train_data = datasets.STL10(osp.join(args.dataset_root, 'STL10'), split='train', transform=train_transform, download=False)
                test_data = datasets.STL10(osp.join(args.dataset_root, 'STL10'), split='test', transform=test_transform, download=False)
            elif args.dataset == 'SVHN':
                train_data = datasets.SVHN(osp.join(args.dataset_root, 'SVHN'), split='train', transform=train_transform, download=False)
                test_data = datasets.SVHN(osp.join(args.dataset_root, 'SVHN'), split='test', transform=test_transform, download=False)
            else:
                raise Exception('Unknown dataset: {}'.format(args.dataset))


            dataloaders = dict()
            dataloaders['train'] = data.DataLoader(train_data,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=args.workers
                                                )

            dataloaders['test'] = data.DataLoader(test_data,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                num_workers=args.workers
                                                )

            if args.model in model_choices:
                model = Network().construct(args.model, config)
            else:
                raise Exception('Unknown model: {}'.format())

            model = model.to(device)

            if args.loss == 'ce':
                criterion = nn.CrossEntropyLoss()
            else:
                raise Exception('Only cross entropy is allowed: {}'.format(args.loss))

            if args.optimizer == 'sgd':
                optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=0.9)
            elif args.optimizer == 'adam':
                optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
            else:
                raise Exception('Optimizer not allowed: {}'.format(args.optimizer))
                
            # scheduler = lr_scheduler.MultiStepLR(optimizer, args.milestones, gamma=args.step_gamma)

            # if args.resume is not None:
            #     assert osp.exists(args.resume)
            #     assert osp.isfile(args.resume)
            #     ckpt = torch.load(args.resume)
            #     assert 'model' in ckpt.keys()
            #     model.load_state_dict(ckpt['model'], strict=True)

            run_acc_list, run_loss_list, trace_list = train(model,
                                                optimizer,
                                                # scheduler,
                                                dataloaders,
                                                criterion,
                                                device,
                                                args.sp,
                                                args.sf,
                                                train_type,
                                                num_epochs=args.num_epochs,
                                                args=args,
                                                ckpt_dir=ckpt_dir,
                                                dataset_sizes=dataset_sizes,
                                                images_dir=images_dir
                                                )

    arr_path = osp.join(ckpt_dir, 'hessian_trace_train_stats.npz')
    kwargs = dict()
    for train_type in ['ord']:
        for phase in ['train', 'test']:
            kwargs['{}_{}_acc'.format(train_type, phase)] = np.array(all_acc[phase][train_type], dtype=np.float32)
            kwargs['{}_{}_loss'.format(train_type, phase)] = np.array(all_loss[phase][train_type], dtype=np.float32)
    kwargs['trace'] = np.array(trace_list,dtype = np.float32)
    np.savez(arr_path, **kwargs)
    logger.info('train stats saved at: {}'.format(arr_path))
