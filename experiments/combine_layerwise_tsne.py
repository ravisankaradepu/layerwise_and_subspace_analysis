#!/usr/bin/env python3

import os
import sys
import argparse
import numpy as np
import os.path as osp

sys.path.append(osp.dirname(os.getcwd()))
from models.cifar import Network
from utils import Config
from PIL import Image, ImageDraw, ImageFont

def parse_args():

    parser = argparse.ArgumentParser(description='combine layerwise and global plots for a run dir')

    model_choices = ['VGG11_bn', 'ResNet18', 'DenseNet3_40']

    parser.add_argument('-r', '--run', type=str, required=True, help='run directory to analyze')
    parser.add_argument('-m', '--model', type=str, choices=model_choices, required=True, help='model that run dir corresponds to')
    parser.add_argument('--pdb', action='store_true', help='run with debugger')
    parser.add_argument('-p', '--perplexity', type=int, required=True, help='perplexity')

    return parser.parse_args()
    

if __name__ == '__main__':

    args = parse_args()
    if args.pdb:
        import pdb
        pdb.set_trace()

    assert osp.exists(args.run), '{} was not found'.format(args.run)
    assert osp.isdir(args.run), '{} is not a directory'.format(args.run)

    images_dir = osp.join(args.run, 'images')
    
    assert osp.exists(images_dir), '{} was not found'.format(images_dir)
    assert osp.isdir(images_dir), '{} is not a directory'.format(images_dir)

    collective_dir = osp.join(images_dir, 'collective')
    if not osp.exists(collective_dir):
        os.makedirs(collective_dir)

    model = Network().construct(args.model, Config())

    text_size = 30
    font = ImageFont.truetype("Roboto-Light.ttf", text_size)
    not_found = list()

    num_layers = 0
    height, width = None, None
    for layer_name, _ in model.named_parameters():
        if height is None:
            dummy_image_path = osp.join(images_dir, '{}_{}p_tsne.png'.format(layer_name, args.perplexity))
            assert osp.exists(dummy_image_path)
            dummy_image = Image.open(dummy_image_path)
            height = dummy_image.height
            width = dummy_image.width
        num_layers += 1

    row_count = int(np.sqrt(num_layers))
    col_count = (num_layers // row_count) if num_layers % row_count == 0 else (num_layers //  row_count + 1)

    print('rows:', row_count)
    print('cols:', col_count)
    print('num_layers:', num_layers)
    tsne_collection = Image.new('RGB', (width * col_count, height * row_count), (255, 255, 255))

    row = 0
    col = 0
    for layer_name, param in model.named_parameters():
        layerwise_tsne_path = osp.join(images_dir, '{}_{}p_tsne.png'.format(layer_name, args.perplexity))
        if not osp.exists(layerwise_tsne_path):
            not_found.append(layerwise_tsne_path)
        layer_tsne_path = Image.open(layerwise_tsne_path)
        tsne_collection.paste(layer_tsne_path, (width * col, height * row))
        if col == col_count - 1:
            row += 1
            col = 0
        else:
            col += 1

    tsne_collection_path = osp.join(images_dir, 'collective_{}p_tsne.png'.format(args.perplexity))
    tsne_collection.save(tsne_collection_path)
    if col_count * row_count == num_layers:
        assert row == row_count
        assert col == 0
    else:
        assert row == row_count - 1
        assert col_count * row + col == num_layers

    print('not found:')
    print(not_found)