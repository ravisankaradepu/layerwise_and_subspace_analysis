#!/usr/bin/env python3

import os
import sys
import torch
import logging
import argparse
import fnmatch
import numpy as np
import pprint as pp
import pandas as pd
import seaborn as sns
import os.path as osp
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torchvision import datasets
from sklearn.manifold import TSNE

sys.path.append(osp.dirname(os.getcwd()))
from models import Network, model_urls, download_model
from hessian import LayerHessian
from utils import get_mean_std, get_subset_dataset, make_deterministic


def parse_args():

    parser = argparse.ArgumentParser(description='layerwise tsne')

    model_choices = ['VGG11_bn', 'ResNet18', 'DenseNet3_40', 'MobileNet', 'LeNet']
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
    parser.add_argument('-l', '--layer', type=str, required=True, help='name of layer to be analyzed')
    parser.add_argument('--print_layers', action='store_true', help='print layer names and exit')
    parser.add_argument('-r', '--run', type=str, required=True, help='run directory to analyze')
    parser.add_argument('-p', '--perplexity', type=int, default=10, help='tsne-perplexity')
    parser.add_argument('--dataset_root', type=str, default=default_dataset_root, help='dataset root, irrespective of dataset type, default={}'.format(default_dataset_root))
    parser.add_argument('--examples_per_class', type=int, default=default_examples_per_class, help='examples per class, default={}'.format(default_examples_per_class))
    parser.add_argument('-b', '--batch_size', type=int, default=default_batch_size, help='batch size, default={}'.format(default_batch_size))
    parser.add_argument('--dpi', type=int, default=default_dpi, help='dpi for saving images, default={}'.format(default_dpi))
    parser.add_argument('--pdb', action='store_true', help='run with debuggger')

    return parser.parse_args()


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
    logger = logging.getLogger('{}_p{}_tsne_decomp'.format(args.layer, args.perplexity))
    logging_file = osp.join(log_dir, '{}_{}p_tsne_decomp.log'.format(args.layer, args.perplexity))
    with open(logging_file, 'w+') as f:
        pass

    logging_file_handler = logging.FileHandler(logging_file)
    logger.addHandler(logging_file_handler)
    logger.info('Arguments: {}'.format(args))
 
    pattern = 'https://storage.googleapis.com/hs-deep-lab-donoho-papyan-bucket/' \
        + 'trained_models/*/*/results/'                                          \
        + 'dataset=' + args.dataset                                              \
        + '-net=' + args.model                                                   \
        + '-lr=*'                                                                \
        + '-examples_per_class=' + str(args.examples_per_class)                  \
        + '-num_classes=*'                                                       \
        + '-epc_seed=*'                                                          \
        + '-train_seed=*'                                                        \
        + '-epoch=*'                                                             \
        + '.pth'

    model_url = fnmatch.filter(model_urls, pattern)[0]
    model_weights_path, results_path = download_model(model_url, ckpt_dir)
    df = pd.read_csv(results_path)
    row = df.iloc[0]

    if args.model in ['VGG11_bn', 'ResNet18', 'DenseNet3_40', 'MobileNet', 'LeNet']:
        model = Network().construct(args.model, row)
    else:
        raise Exception('Unknown model argument: {}'.format(args.model))

    if args.print_layers:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('Total number of parameters: ', total_params, '\nlayers:')
        pp.pprint(list(name for (name, _) in model.named_parameters()))
        exit(code=0)

    state_dict = torch.load(model_weights_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    if args.cuda is None:
        gpus = torch.cuda.device_count()
        if gpus > 1:
            model = nn.DataParallel(model, device_ids=range(gpus))
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

    C = row.num_classes
    logger.info('Starting G decomposition...')
    res = LayerHessian(crit='CrossEntropyLoss',
                loader=loader,
                device=device,
                model=model,
                layer_name=args.layer,
                num_classes=C,
                hessian_type='G',
                ).compute_G_decomp()

    logger.info('Starting TSNE...')
    tsne_embedded = TSNE(n_components=2,
                        metric='precomputed',
                        perplexity=args.perplexity).fit_transform(res['dist'])
    logger.info('TSNE finished...')
    # t-SNE X
    delta_c_X = tsne_embedded[:C,0]
    delta_ccp_X = tsne_embedded[C:,0]

    # t-SNE Y
    delta_c_Y = tsne_embedded[:C,1]
    delta_ccp_Y = tsne_embedded[C:,1]

    # True class c is in the first entry of res['labels']
    c = [x[0] for x in res['labels']]
    delta_c_C = c[:C]
    delta_ccp_C = c[C:]

    logger.info('Creating plot...')
    plt.figure()

    plt.scatter(delta_c_X,   delta_c_Y,   c=delta_c_C,   s=300, cmap=plt.get_cmap('rainbow'), label='cluster centers $\delta_c$', alpha=0.5)
    plt.scatter(delta_ccp_X, delta_ccp_Y, c=delta_ccp_C, s=10,  cmap=plt.get_cmap('rainbow'), label="cluster members $\delta_{c,c'}$")

    plt.xlabel('t-SNE X')
    plt.ylabel('t-SNE Y')

    plt.legend()

    image_path = osp.join(images_dir, '{}_{}p_tsne.png'.format(args.layer, args.perplexity))
    plt.savefig(image_path, cpi=args.dpi)
    logger.info('Saved plot...')