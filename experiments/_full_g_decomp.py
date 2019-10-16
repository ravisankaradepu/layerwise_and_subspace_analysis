#!/usr/bin/env python3

import os
import sys
import torch
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
from models.cifar import ResNet18
from hessian import FullHessian
from utils import get_mean_std, get_subset_dataset, make_deterministic, Config


def parse_args():

    parser = argparse.ArgumentParser(description='full g decomposition')

    model_choices = ['VGG11_bn', 'ResNet18', 'DenseNet3_40']
    dataset_choices = ['CIFAR10', 'CIFAR100', 'MNIST', 'FashionMNIST']

    default_model = model_choices[0]
    default_dataset = dataset_choices[0]
    default_dataset_root = osp.join(osp.dirname(os.getcwd()), 'datasets')
    default_examples_per_class = 10
    default_batch_size = 1024
    default_dpi = 600

    parser.add_argument('--cuda', type=int, help='number of gpu to be used, default=None')
    parser.add_argument('-d', '--dataset', type=str, default=default_dataset, choices=dataset_choices, help='dataset to analyze, default={}'.format(default_dataset))
    parser.add_argument('-m', '--model', type=str, default=default_model, choices=model_choices, help='model for analysis, default={}'.format(default_model))
    parser.add_argument('-r', '--run', type=str, required=True, help='run directory to analyze')
    parser.add_argument('--dataset_root', type=str, default=default_dataset_root, help='dataset root, irrespective of dataset type, default={}'.format(default_dataset_root))
    parser.add_argument('--examples_per_class', type=int, default=default_examples_per_class, help='examples per class, default={}'.format(default_examples_per_class))
    parser.add_argument('-b', '--batch_size', type=int, default=default_batch_size, help='batch size, default={}'.format(default_batch_size))
    parser.add_argument('--dpi', type=int, default=default_dpi, help='dpi for saving images, default={}'.format(default_dpi))
    parser.add_argument('--pdb', action='store_true', help='run with debuggger')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    if args.pdb is None:
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
    logger = logging.getLogger('full_g_decomp')
    logging_file = osp.join(log_dir, 'full_g_decomp.log')
    with open(logging_file, 'w+') as f:
        pass

    logging_file_handler = logging.FileHandler(logging_file)
    logger.addHandler(logging_file_handler)
    logger.info('Arguments: {}'.format(args))
 
    # pattern = 'https://storage.googleapis.com/hs-deep-lab-donoho-papyan-bucket/' \
    #     + 'trained_models/*/*/results/'                                          \
    #     + 'dataset=' + args.dataset                                              \
    #     + '-net=' + args.model                                                   \
    #     + '-lr=*'                                                                \
    #     + '-examples_per_class=' + str(args.examples_per_class)                  \
    #     + '-num_classes=*'                                                       \
    #     + '-epc_seed=*'                                                          \
    #     + '-train_seed=*'                                                        \
    #     + '-epoch=*'                                                             \
    #     + '.pth'

    # model_url = fnmatch.filter(model_urls, pattern)[0]
    # model_weights_path, results_path = download_model(model_url, ckpt_dir)
    # df = pd.read_csv(results_path)
    # row = df.iloc[0]
    model_weights_path = osp.join(ckpt_dir, 'model_weights.pth')
    if args.dataset in ['MNIST', 'FashionMNIST']:
        input_ch = 1
        padded_im_size = 28
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
    if args.model in ['ResNet18', 'DenseNet3_40', 'VGG11_bn']:
        model = Network().construct(args.model, row)
    else:
        raise Exception('Unknown model argument: {}'.format(args.model))

    state_dict = torch.load(model_weights_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict['model'], strict=True)
    model = model.to(device)

    model = model.eval()

    mean, std = get_mean_std(args.dataset)
    pad = int((row.padded_im_size-row.im_size)/2)
    transform = transforms.Compose([transforms.Pad(pad),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean,std)])

    full_dataset = getattr(datasets, args.dataset)
    subset_dataset = get_subset_dataset(full_dataset=full_dataset,
                                        examples_per_class=args.examples_per_class,
                                        epc_seed=row.epc_seed,
                                        root=osp.join(args.dataset_root, args.dataset),
                                        train=True,
                                        transform=transform,
                                        download=True
                                        )

    loader = DataLoader(dataset=subset_dataset,
                        drop_last=False,
                        batch_size=args.batch_size)

    logger.info('Starting G decomposition...')
    res = FullHessian(crit='CrossEntropyLoss',
                  loader=loader,
                  device=device,
                  model=model,
                  num_classes=row.num_classes,
                  hessian_type='G',
                  ).compute_G_decomp()

    logger.info('G decomposition finished...')
    logger.info('Starting G eigenspectrum computation')
    Hess = FullHessian(crit='CrossEntropyLoss',
                   loader=loader,
                   device=device,
                   model=model,
                   num_classes=row.num_classes,
                   hessian_type='G',
                   init_poly_deg=64,     # number of iterations used to compute maximal/minimal eigenvalue
                   poly_deg=128,         # the higher the parameter the better the approximation
                   spectrum_margin=0.05,
                   poly_points=1024,     # number of points in spectrum approximation
                   SSI_iters=128         # iterations of subspace iterations
                   )

    eigval, eigval_density = Hess.LanczosLoop(denormalize=True)
    logger.info('G eigenspectrum computation finished')
    logger.info('Creating plots...')
    plt.figure()

    plt.semilogy(eigval, eigval_density)

    plt.scatter(res['G0_eigval'],  res['G0_eigval_density'],  c='cyan',   label='$G_0$')
    plt.scatter(res['G1_eigval'],  res['G1_eigval_density'],  c='orange', label='$G_1$')
    plt.scatter(res['G2_eigval'],  res['G2_eigval_density'],  c='red',    label='$G_2$')
    plt.scatter(res['G12_eigval'], res['G12_eigval_density'], c='green',  label='$G_{1+2}$')

    plt.legend()

    image_path = osp.join(images_dir, 'full_g_decomp.png')
    plt.savefig(image_path, cpi=args.dpi)
    logger.info('Saved points...')