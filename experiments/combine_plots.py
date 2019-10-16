#!/usr/bin/env python3

import os
import sys
import argparse
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

    full_hessian_decomp_path = osp.join(images_dir, 'full_hessian_decomp.png')
    full_g_decomp_path = osp.join(images_dir, 'full_g_decomp.png')
    full_tsne_path = osp.join(images_dir, 'full_tsne.png')

    assert osp.exists(full_hessian_decomp_path), '{} was not found'.format(full_hessian_decomp_path)
    assert osp.isfile(full_hessian_decomp_path), '{} is not a file'.format(full_hessian_decomp_path)

    assert osp.exists(full_g_decomp_path), '{} was not found'.format(full_g_decomp_path)
    assert osp.isfile(full_g_decomp_path), '{} is not a file'.format(full_g_decomp_path)

    assert osp.exists(full_tsne_path), '{} was not found'.format(full_tsne)
    assert osp.isfile(full_tsne_path), '{} is not a file'.format(full_tsne)

    top_margin = 50
    left_margin = 100
    text_size = 30
    font = ImageFont.truetype("Roboto-Light.ttf", text_size)
    not_found = list()

    for layer_name, param in model.named_parameters():
        
        layer_g_decomp_path = osp.join(images_dir, '{}_g_decomp.png'.format(layer_name))
        layer_tsne_path = osp.join(images_dir, '{}_tsne.png'.format(layer_name))
        layer_hessian_path = osp.join(images_dir, '{}_hessian_decomp.png'.format(layer_name))

        layer_collection_image_path = osp.join(collective_dir, '{}_collection.png'.format(layer_name))

        # assert osp.exists(layer_tsne_path), '{} was not found'.format(layer_name)
        # assert osp.isfile(layer_tsne_path), '{} is not a file'.format(layer_name)

        # assert osp.exists(layer_g_decomp_path), '{} was not found'.format(layer_name)
        # assert osp.isfile(layer_g_decomp_path), '{} is not a file'.format(layer_name)
        if not osp.exists(layer_tsne_path):
            not_found.append(layer_tsne_path)
            continue
        if not osp.exists(layer_g_decomp_path):
            not_found.append(layer_g_decomp_path)
            continue
        if not osp.exists(layer_hessian_path):
            not_found.append(layer_hessian_path)
            continue

        full_hessian_decomp = Image.open(full_hessian_decomp_path)
        full_g_decomp = Image.open(full_g_decomp_path)
        full_tsne = Image.open(full_tsne_path)
        layer_g_decomp = Image.open(layer_g_decomp_path)
        layer_tsne = Image.open(layer_tsne_path)
        layer_hessian = Image.open(layer_hessian_path)

        layer_collection = Image.new('RGB', (left_margin + full_hessian_decomp.width * 3, top_margin + full_g_decomp.height + layer_g_decomp.height), (255, 255, 255))
        
        draw = ImageDraw.Draw(layer_collection)

        draw.text((left_margin * 0.1, top_margin * 0.1), 'layer: ' + layer_name, (0, 0, 0), size=text_size, font=font)
        draw.text((left_margin * 0.1, top_margin * 0.6), 'shape: {}'.format(param.shape))
        draw.text((left_margin * 0.1, top_margin + full_g_decomp.height // 2), 'FULL', (0, 0, 0), sise=text_size, font=font)
        draw.text((left_margin * 0.1, top_margin + full_g_decomp.height + layer_g_decomp.height // 2), 'LAYER', (0, 0, 0), size=text_size, font=font)
        draw.text((left_margin + full_g_decomp.width // 2, top_margin * 0.4), 'G-decomposition', (0, 0, 0), size=text_size, font=font)
        draw.text((left_margin + full_g_decomp.width + full_tsne.width // 2, top_margin * 0.4), 'T-SNE', (0, 0, 0), size=text_size, font=font)
        draw.text((left_margin + full_g_decomp.width + full_tsne.width + full_hessian_decomp.width // 3, top_margin * 0.4), 'Hessian-decomposition', (0, 0, 0), size=text_size, font=font)


        layer_collection.paste(full_g_decomp, (left_margin, top_margin))
        layer_collection.paste(full_tsne, (left_margin + full_g_decomp.width, top_margin))
        layer_collection.paste(full_hessian_decomp, (left_margin + full_g_decomp.width + full_tsne.width, top_margin))
        layer_collection.paste(layer_g_decomp, (left_margin, top_margin + full_g_decomp.height))
        layer_collection.paste(layer_tsne, (left_margin + layer_g_decomp.width, top_margin + full_tsne.height))
        layer_collection.paste(layer_hessian, (left_margin + full_g_decomp.width + full_tsne.width, top_margin + full_hessian_decomp.height))
        layer_collection.save(layer_collection_image_path)

    print('not found:')
    print(not_found)