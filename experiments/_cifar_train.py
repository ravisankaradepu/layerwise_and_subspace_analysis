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
import torch.utils.data as data
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.datasets as datasets
import torchvision.transforms as transforms
sys.path.append(osp.dirname(os.getcwd()))

from models.cifar import ResNet18, Network
from utils import get_mean_std, Config


def parse_args():

    parser = argparse.ArgumentParser(description='train on mnist with single level model')

    default_learning_rate = 1e-4
    default_l2 = 0.0
    default_num_epochs = 20
    default_dataset = 'CIFAR10'
    default_batch_size = 32
    default_workers = 4
    default_model = 'vgg11'
    default_milestone = [10]
    default_step_gamma = 0.1

    dataset_choices = ['CIFAR10', 'CIFAR100']
    model_choices = ['vgg16', 'resnet18', 'densenet40']
    optimizers_available = ['sgd']
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

    parser.add_argument('--num_epochs',
                        type=int,
                        default=default_num_epochs,
                        help='number of training epochs, default={}'.format(default_num_epochs)
                        )

    parser.add_argument('-o',
                        '--optimizer',
                        type=str,
                        required=True,
                        choices=optimizers_available,
                        help='optimizer'
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

    parser.add_argument('--batch_size',
                        type=int,
                        default=default_batch_size,
                        help='batch size for training, default={}'.format(default_batch_size)
                        )

    parser.add_argument('--workers',
                        type=int,
                        default=default_workers,
                        help='number of wrokers for dataloader, default={}'.format(default_workers)
                        )

    parser.add_argument('--dataset_root',
                        type=str,
                        default=osp.join(osp.dirname(os.getcwd()), 'datasets'),
                        help='directory for dataset, default={}'.format(osp.join(osp.dirname(os.getcwd()), 'datasets'))
                        )

    parser.add_argument('--log_dir',
                        type=str,
                        default='logs',
                        help='directory for logs, default=./logs'
                        )

    parser.add_argument('--ckpt_dir',
                        type=str,
                        default='ckpt',
                        help='directory to store checkpoints, '
                             'default=./ckpt'
                        )

    parser.add_argument('--images_dir',
                        type=str,
                        default='images',
                        help='directory to store images'
                             ', default=./images'
                        )

    parser.add_argument('--model',
                        type=str,
                        default=default_model,
                        choices=model_choices,
                        help='model type, default={}'.format(default_model)
                        )

    parser.add_argument('--use_cuda',
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

    parser.add_argument('--save_each_epoch',
                        action='store_true',
                        help='save weights at each epoch end'
                        )

    parser.add_argument('--milestones',
                        type=int,
                        nargs='+',
                        default=default_milestone,
                        help='milestones for multistep-lr scheduler, '
                        'default={}'.format(default_milestone)
                        )

    parser.add_argument('--step_gamma',
                        type=float,
                        default=default_step_gamma,
                        help='gamma for step-lr scheduler'
                        ', default={}'.format(default_step_gamma)
                        )

    parser.add_argument('--augment',
                        action='store_true',
                        help='augment data with random-flip and random crop'
                        )

    parser.add_argument('--resume',
                        type=str,
                        help='path to ckpt.pth to resume training'
                        )

    return parser.parse_args()

def evaluate_model(model, criterion, dataloader, device, dataset_size):

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

            loss = criterion(output, preds)
            running_loss += loss.item() * batch.size(0)
    return {'loss': running_loss / dataset_size, 'acc': running_corrects.double() / dataset_size}


def train(model,
          optimizer,
          scheduler,
          dataloaders,
          criterion,
          device,
          num_epochs=100,
          args=None,
          dataset_sizes={'train': 5e4, 'test': 1e4},
          images_dir=None,
          ckpt_dir=None
          ):

    logger = logging.getLogger('train')
    loss_list = {'train': list(), 'test': list()}
    acc_list = {'train': list(), 'test': list()}

    assert images_dir is not None
    assert ckpt_dir is not None

    loss_image_path = osp.join(images_dir, 'loss.png')
    acc_image_path = osp.join(images_dir, 'acc.png')

    model.train()
    for epoch in range(num_epochs):
        logger.info('epoch: %d' % epoch)
        with torch.enable_grad():
            for batch, truth in dataloaders['train']:

                batch = batch.to(device)
                truth = truth.to(device)
                optimizer.zero_grad()

                output = model(batch)
                loss = criterion(output, truth)

                loss.backward()
                optimizer.step()
                scheduler.step()

        for phase in ['train', 'test']:

            stats = evaluate_model(model, criterion, dataloaders[phase], device, dataset_sizes[phase])

            loss_list[phase].append(stats['loss'])
            acc_list[phase].append(stats['acc'])

            logger.info('{}:'.format(phase))
            logger.info('\tloss:{}'.format(stats['loss']))
            logger.info('\tacc :{}'.format(stats['acc']))

            if phase == 'test':
                plt.clf()
                plt.plot(loss_list['test'], label='test_loss')
                plt.plot(loss_list['train'], label='train_loss')
                plt.legend()
                plt.savefig(loss_image_path)

                plt.clf()
                plt.plot(acc_list['test'], label='test_acc')
                plt.plot(acc_list['train'], label='train_acc')
                plt.legend()
                plt.savefig(acc_image_path)
                plt.clf()

        if args.save_each_epoch:
            current_system = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}

            epoch_weights_path = osp.join(ckpt_dir, 'model_weights_epochs_{}.pth'.format(epoch))
            torch.save(current_system, epoch_weights_path)

    return {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}


if __name__ == '__main__':

    args = parse_args()
    if args.with_pdb:
        import pdb
        pdb.set_trace()
    np.random.seed(0)
    torch.manual_seed(0)
    if args.use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    sns.set_style('darkgrid')
    if args.use_cuda is None:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:%d' % args.use_cuda)
        cudnn.benchmark = True
    log_dir = osp.join(args.run, 'logs')
    ckpt_dir = osp.join(args.run, 'ckpt')
    images_dir = osp.join(args.run, 'images')
    if not osp.exists(args.run):
        os.makedirs(args.run)
        if not osp.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        if not osp.exists(images_dir):
            os.makedirs(images_dir)
        if not osp.exists(log_dir):
            os.makedirs(log_dir)

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logging_file = osp.join(log_dir, 'train.log')
    logger = logging.getLogger('train')
    with open(logging_file, 'w+') as f:
        pass
    logger_file_handler = logging.FileHandler(logging_file)
    logger.addHandler(logger_file_handler)
    logger.info('Arguments: {}'.format(args))

    mean, std = get_mean_std(args.dataset)
    train_transform = transforms.Compose([transforms.Pad(0),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean,std)])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_data = datasets.CIFAR10(osp.join(args.dataset_root, 'CIFAR10'), transform=train_transform, train=True, download=False)
    test_data = datasets.CIFAR10(osp.join(args.dataset_root, 'CIFAR10'), transform=test_transform, train=False, download=False)

    dataloaders = dict()
    dataloaders['train'] = data.DataLoader(train_data,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           num_workers=args.workers
                                           )

    dataloaders['test'] = data.DataLoader(test_data,
                                          batch_size=args.batch_size,
                                          shuffle=True,
                                          num_workers=args.workers
                                          )

    if args.dataset == 'cifar100':
        info.num_classes = 100
    if args.model == 'efficient':
        model = EfficientNetB0()
    elif args.model.startswith('vgg'):
        model = VGG(args.model)
    elif args.model == 'resnet18':
        model = Network().construct('ResNet18', Config())
    elif args.model == 'resnet34':
        model = ResNet34(num_classes=10)
    elif args.model == 'resnet50':
        model = ResNet50()
    elif args.model == 'resnet101':
        model = ResNet101()
    elif args.model == 'densenet121':
        model = DenseNet121()
    elif args.model == 'mobilenet':
        model = MobileNet()
    elif args.model == 'mobilenetv2':
        model = MobileNetV2()
    elif args.model == 'lenet':
        model = LeNet()
    elif args.model == 'densenet40':
        model = DenseNet40(info)
    else:
        raise Exception('Unknown model: {}'.format())

    model = model.to(device)
    # if args.use_cuda is not None:
    #     model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=0.9)
    # optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=0.9)
    scheduler = lr_scheduler.MultiStepLR(optimizer, args.milestones, gamma=args.step_gamma)

    if args.resume is not None:
        assert osp.exists(args.resume)
        assert osp.isfile(args.resume)
        ckpt = torch.load(args.resume)
        assert 'model' in ckpt.keys()
        model.load_state_dict(ckpt['model'])

    system = train(model,
                   optimizer,
                   scheduler,
                   dataloaders,
                   criterion,
                   device,
                   num_epochs=args.num_epochs,
                   args=args,
                   ckpt_dir=ckpt_dir,
                   images_dir=images_dir
                   )

    torch.save(system, osp.join(ckpt_dir, 'model_weights.pth'))