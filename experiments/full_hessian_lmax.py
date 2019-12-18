#!/usr/bin/env python3

import os
import sys
import glob
import torch
import pickle
import logging
import argparse
import fnmatch
import numpy as np
import pandas as pd
import seaborn as sns
import os.path as osp
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torchvision import datasets

sys.path.append(osp.dirname(os.getcwd()))
from models import Network, model_urls, download_model
from hessian import FullHessian
from utils import get_mean_std, get_subset_dataset, make_deterministic, Config


def parse_args():

    parser = argparse.ArgumentParser(description='layerwise hessian dynamic decomposition')

    model_choices = ['VGG11', 'VGG11_bn', 'VGG13', 'VGG13_bn', 'VGG16', 'VGG16_bn', 'VGG19', 'VGG19_bn', 'ResNet18', 'DenseNet3_40', 'MobileNet', 'LeNet']
    dataset_choices = ['CIFAR10', 'CIFAR100', 'MNIST', 'FashionMNIST', 'STL10']

    default_model = model_choices[0]
    default_dataset = dataset_choices[0]
    default_dataset_root = osp.join(osp.dirname(os.getcwd()), 'datasets')
    default_examples_per_class = 10
    default_batch_size = 1024
    default_dpi = 600

    parser.add_argument('--cuda', type=int, help='number of gpu to be used, default=None')
    parser.add_argument('-d', '--dataset', type=str, choices=dataset_choices, help='dataset to analyze, default={}'.format(default_dataset))
    parser.add_argument('-m', '--model', type=str, choices=model_choices, help='model for analysis, default={}'.format(default_model))
    parser.add_argument('-r', '--run', type=str, required=True, help='run directory to analyze')
    parser.add_argument('--dataset_root', type=str, default=default_dataset_root, help='dataset root, irrespective of dataset type, default={}'.format(default_dataset_root))
    parser.add_argument('--examples_per_class', type=int, default=default_examples_per_class, help='examples per class, default={}'.format(default_examples_per_class))
    parser.add_argument('-b', '--batch_size', type=int, default=default_batch_size, help='batch size, default={}'.format(default_batch_size))
    parser.add_argument('--dpi', type=int, default=default_dpi, help='dpi for saving images, default={}'.format(default_dpi))
    parser.add_argument('--pdb', action='store_true', help='run with debuggger')
    parser.add_argument('--num', type=int, help = 'numer of models in ckpt dir')

    return parser.parse_args()


def get_model_weight_paths(ckpt_dir,num):

#    model_weights_paths = sorted(glob.glob(osp.join(ckpt_dir, '*.pth')))
    model_weights_paths = list()
    for i in range(num):
        s=osp.join(ckpt_dir,'model_weights_epochs_'+str(i)+'.pth')
        model_weights_paths.append(s)	

    epochs_paths = list()
    for model_weight_path in model_weights_paths:
        assert osp.exists(model_weight_path)
        # file_name = model_weight_path[len(ckpt_dir):].lstrip('/')
        if 'model_weights_epochs' in model_weight_path:
            epoch_num = int(model_weight_path.split('_')[-1][:-len('.pth')])
        else:
            epoch_num = 'inf'

        epochs_paths.append((epoch_num, model_weight_path))
    
    inf_entry = epochs_paths[0]
    epochs_paths = epochs_paths[1:]
    epochs_paths.append(inf_entry)

    return epochs_paths


if __name__ == '__main__':

    args = parse_args()
    if args.pdb:
        import pdb
        pdb.set_trace()
    make_deterministic(args.cuda)
    sns.set_style('darkgrid')
    device = torch.device('cpu' if args.cuda is None else 'cuda:{}'.format(args.cuda))

    if not osp.exists(args.run):
        os.makedirs(args.run)

    ckpt_dir = osp.join(args.run, 'ckpt')
    images_dir = osp.join(args.run, 'images')
    log_dir = osp.join(args.run, 'logs')

    if not osp.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    if not osp.exists(images_dir):
        os.makedirs(images_dir)
    if not osp.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(level=logging.INFO,  format='%(message)s')
    logger = logging.getLogger('full_hessian_dyn_lmax')
    logger_file_name = 'full_hessian_dyn_lmax.log'

    logging_file = osp.join(log_dir, logger_file_name)
    with open(logging_file, 'w+') as f:
        pass

    logging_file_handler = logging.FileHandler(logging_file)
    logger.addHandler(logging_file_handler)
    logger.info('Arguments: {}'.format(args))

    # assert osp.exists(model_weights_path), '{} was not found'.format(model_weights_path)
    # assert osp.isfile(model_weights_path), '{} is not a file'.format(model_weights_path)
    if args.dataset in ['MNIST', 'FashionMNIST']:
        input_ch = 1
        padded_im_size = 32
        num_classes = 10
        im_size = 28
        epc_seed = 0
        row = Config(input_ch=input_ch, 
                    padded_im_size=padded_im_size, 
                    num_classes=num_classes,
                    im_size=im_size,
                    epc_seed=epc_seed
                    )
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
        row = Config(input_ch=input_ch, 
                    padded_im_size=padded_im_size,
                    num_classes=num_classes,
                    im_size=im_size,
                    epc_seed=epc_seed
                    )
    elif args.dataset in ['STL10']:
        input_ch = 3
        padded_im_size = 101
        num_classes = 10
        im_size = 96
        epc_seed = 0
        row = Config(input_ch=input_ch,
                        padded_im_size=padded_im_size,
                        num_classes=num_classes,
                        im_size=im_size,
                        epc_seed=epc_seed
                        )
    else:
        raise Exception('this was expected to be an unreachable line')        

    if args.model in ['VGG11', 'VGG11_bn', 'VGG13', 'VGG13_bn', 'VGG16', 'VGG16_bn', 'VGG19', 'VGG19_bn', 'ResNet18', 'DenseNet3_40', 'MobileNet', 'LeNet']:
        model = Network().construct(args.model, row)
    else:
        raise Exception('Unknown model argument: {}'.format(args.model))

    # state_dict = torch.load(model_weights_path, map_location=lambda storage, loc: storage)
    # if 'model' in state_dict.keys():
    #     state_dict = state_dict['model']
    # model.load_state_dict(state_dict, strict=True)
    # model = model.to(device)

    # model = model.eval()

    mean, std = get_mean_std(args.dataset)
    pad = int((row.padded_im_size-row.im_size)/2)
    transform = transforms.Compose([transforms.Pad(pad),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean,std)])
    if args.dataset in ['MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100']:
        full_dataset = getattr(datasets, args.dataset)
        subset_dataset = get_subset_dataset(full_dataset=full_dataset,
                                            examples_per_class=args.examples_per_class,
                                            epc_seed=row.epc_seed,
                                            root=osp.join(args.dataset_root, args.dataset),
                                            train=True,
                                            transform=transform,
                                            download=True
                                            )
    elif args.dataset == 'STL10':
        full_dataset = datasets.STL10
        subset_dataset = get_subset_dataset(full_dataset=full_dataset,
                                            examples_per_class=args.examples_per_class,
                                            epc_seed=row.epc_seed,
                                            root=osp.join(args.dataset_root, args.dataset),
                                            split='train',
                                            transform=transform,
                                            download=False
                                            )
    else:
        raise Exception('Unknown dataset: {}'.format(args.dataset))
    loader = DataLoader(dataset=subset_dataset,
                        drop_last=False,
                        batch_size=args.batch_size)

    C = row.num_classes

    lmax_list, lmin_list = list(), list()
    logger.info('Starting full hessian decomposition...')

    model_weight_paths = get_model_weight_paths(ckpt_dir,args.num)

    for epoch_index, (epoch_number, weights_path) in enumerate(model_weight_paths):

        logger.info('Starting epoch: {}'.format(epoch_number))

        assert osp.exists(weights_path), 'path to weights: {} was not found'.format(weights_path)
        state_dict = torch.load(weights_path, map_location=lambda storage, loc: storage)
        if 'model' in state_dict.keys():
            state_dict = state_dict['model']
        model.load_state_dict(state_dict, strict=True)
        model = model.to(device)

        model = model.eval()
        logger.info('weights loaded from path: {}'.format(weights_path))
        logger.info('for epoch: {}'.format(epoch_number))

        Hess = FullHessian(crit='CrossEntropyLoss',
                            loader=loader,
                            device=device,
                            model=model,
                            num_classes=C,
                            hessian_type='Hessian',
                            init_poly_deg=64,
                            poly_deg=128,
                            spectrum_margin=0.05,
                            poly_points=1024,
                            SSI_iters=128
                            )

        lmin, lmax = Hess.compute_lb_ub()

        lmax_list.append((epoch_number, lmax))
        lmin_list.append((epoch_number, lmin))

        logger.info('computation finished for epoch number: {}'.format(epoch_number))
        print(lmax)
    lmax_path = osp.join(ckpt_dir, 'full_lambda_max.pkl')
    with open(lmax_path, 'w+b') as f:
        pickle.dump({'lmax': lmax_list, 'lmin': lmin_list}, f)

    logger.info('Saved lmax at: {}'.format(lmax_path))

    # plt.figure()
    # plt.plot([l[1] for l in lmax_list])
    # plt.xticks(np.arange(len(lmax_list)), [l[0] for l in lmax_list])
    # # plt.yscale('log')
    # plt.legend()
    # plt.xlabel('epochs')
    # plt.ylabel('full-network $\lambda_{max}$')
    # plt.tight_layout()

    # image_path = osp.join(images_dir, 'full_hessian_lmax.png')

    # plt.savefig(image_path, dpi=args.dpi)
    # logger.info('full hessian lmax finished for model:{}, dataset: {}'.format(args.model, args.dataset))
