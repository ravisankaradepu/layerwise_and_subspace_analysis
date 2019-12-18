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
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import math
from scipy.linalg import subspace_angles

sys.path.append(osp.dirname(os.getcwd()))
from hessian import FullHessian
from models.cifar import Network
from utils import Config, get_mean_std, get_subset_dataset


def parse_args():

    parser = argparse.ArgumentParser(description='train')

    dataset_choices = ['CIFAR10', 'CIFAR100', 'MNIST', 'FashionMNIST']
    model_choices = ['VGG11_bn', 'ResNet18', 'DenseNet3_40', 'LeNet', 'MobileNet']
    optimizer_choices = ['sgd', 'adam']

    default_learning_rate = 1e-4
    default_l2 = 0.0
    default_num_epochs = 100
    default_dataset = dataset_choices[0]
    default_batch_size = 256
    default_workers = 4
    default_model = model_choices[0]
    default_milestone = [30, 60]
    default_step_gamma = 0.1
    default_dataset_root = osp.join(osp.dirname(os.getcwd()) ,'datasets')
    default_log_dir = 'log'
    default_ckpt_dir = 'ckpt'
    default_images_dir = 'images'

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
                        choices=['sgd', 'adam'],
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

    parser.add_argument('--model',
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
                        help='save epoch weights with these freq'
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
                        help='path to *.pth to resume training'
                        )
    parser.add_argument('--num_models',
                        type=int,
                        help='No of models to consider')
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

        if args.save_freq is not None and epoch % args.save_freq == 0:
            # current_system = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}

            epoch_weights_path = osp.join(ckpt_dir, 'model_weights_epochs_{}.pth'.format(epoch))
            torch.save(model.state_dict(), epoch_weights_path)

    return {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}


if __name__ == '__main__':

    args = parse_args()
    if args.with_pdb:
        import pdb
        pdb.set_trace()
    np.random.seed(0)
    torch.manual_seed(0)
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
                    im_size=im_size,
                    epc_seed=epc_seed
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
                    im_size=im_size,
                    epc_seed=epc_seed
                    )
        dataset_sizes = {'train': 5e4, 'test': 1e4}
    else:
        raise Exception('Should not have reached here')

    if args.augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(config.padded_im_size, padding=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        train_transform = transforms.Compose([transforms.Pad(int((config.padded_im_size - config.im_size) / 2)),
            transforms.ToTensor(), 
            transforms.Normalize(mean, std)])
    test_transform = transforms.Compose([transforms.Pad((config.padded_im_size - config.im_size) // 2), transforms.ToTensor(), 
            transforms.Normalize(mean, std)])
    full_dataset = getattr(datasets, args.dataset)
    subset_dataset = get_subset_dataset(full_dataset=full_dataset,
                                        examples_per_class=40,
                                        epc_seed=config.epc_seed,
                                        root=osp.join(args.dataset_root, args.dataset),
                                        train=True,
                                        transform=test_transform,
                                        download=True
                                        )

    loader = DataLoader(dataset=subset_dataset,
                        drop_last=False,
                        batch_size=args.batch_size)

    if args.model in ['VGG11_bn', 'ResNet18', 'DenseNet3_40', 'LeNet', 'MobileNet']:
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
        
    scheduler = lr_scheduler.MultiStepLR(optimizer, args.milestones, gamma=args.step_gamma)
# Modified code from here to calcuate top subspace of Hessian at every iterate
    for i in range(args.num_models):
        model_weights_path = osp.join(ckpt_dir, f'model_weights_epochs_{i}.pth')
        print(f'Processing model number {i}')
        state_dict = torch.load(model_weights_path,device)
        model.load_state_dict(state_dict, strict=True)
        model = model.to(device)
        C = 10
        Hess = FullHessian(crit='CrossEntropyLoss',
                    loader=loader,
                    device=device,
                    model=model,
                    num_classes=C,
                    hessian_type='Hessian',
                    init_poly_deg=64,     # number of iterations used to compute maximal/minimal eigenvalue
                    poly_deg=128,         # the higher the parameter the better the approximation
                    spectrum_margin=0.05,
                    poly_points=1024,     # number of points in spectrum approximation
                    SSI_iters=128,        # iterations of subspace iterations
                    )


# Exact outliers computation using subspace iteration
        eigvecs_cur, eigvals_cur,_ = Hess.SubspaceIteration()

# Flatten out eigvecs_cur as it is a list
        top_subspace = torch.zeros(0,device=device)
        for _ in range(len(eigvecs_cur)):
            b = torch.zeros(0,device=device)
            for j in range(len(eigvecs_cur[0])):
                b=torch.cat((b,torch.flatten(eigvecs_cur[0][j])),0)
            b.unsqueeze_(1)
            top_subspace=torch.cat((top_subspace,b),1)
        eigvecs_cur = top_subspace
                
#       Statistics of subspaces, (1) Angle between top subpaces
        if i != 0: # Skipping the first iteration as we do not have notion of previous hessian
            ang=torch.norm(torch.mm(eigvecs_prev.transpose(0,1), eigvecs_cur),1)
            ang_sb.append(ang)

#    Calculating principal angles
            u,s,v = torch.svd(torch.mm(eigvecs_cur.transpose(0, 1), eigvecs_prev))
            #    Output in radians
            s = torch.acos(torch.clamp(s,min=-1,max=1))
            s = s*180/math.pi
#    Attach 's' to p_angles
            if np.size(p_angles) == 0:
                p_angles = s.detach().cpu().numpy()
                p_angles = np.expand_dims(p_angles, axis=0)
            else:
                p_angles = np.concatenate((p_angles,np.expand_dims(s.detach().cpu().numpy(),axis=0)),0)

        eigvecs_prev = eigvecs_cur
        eigvals_prev = eigvals_cur

# If first model, no p_angles

        save_path = osp.join(args.run, 'subspace/model_{}.npz'.format(i))
        if i == 0:
            np.savez(save_path, eigvecs_cur = eigvecs_cur.cpu().numpy(), eigvals_cur = torch.tensor(eigvals_cur).numpy())
        else:
            np.savez(save_path, eigvecs_cur = eigvecs_cur.cpu().numpy(), eigvals_cur = 
                                                torch.tensor(eigvals_cur).numpy())
# Saving p-angles    
    save_path = osp.join(args.run, 'subspace/Angles.npz')
    np.savez(save_path,ang_sb= torch.tensor(ang_sb).numpy(), p_angles=p_angles)

