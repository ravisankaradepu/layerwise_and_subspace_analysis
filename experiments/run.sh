#!/bin/bash
########################################################################################################
# layerwise tsne for ResNet18+CIFAR10 
########################################################################################################
for layer_name in 'conv1.weight' \
 'bn1.weight' \
 'bn1.bias' \
 'layer1.0.conv1.weight' \
 'layer1.0.bn1.weight' \
 'layer1.0.bn1.bias' \
 'layer1.0.conv2.weight' \
 'layer1.0.bn2.weight' \
 'layer1.0.bn2.bias' \
 'layer1.1.conv1.weight' \
 'layer1.1.bn1.weight' \
 'layer1.1.bn1.bias' \
 'layer1.1.conv2.weight' \
 'layer1.1.bn2.weight' \
 'layer1.1.bn2.bias' \
 'layer2.0.conv1.weight' \
 'layer2.0.bn1.weight' \
 'layer2.0.bn1.bias' \
 'layer2.0.conv2.weight' \
 'layer2.0.bn2.weight' \
 'layer2.0.bn2.bias' \
 'layer2.0.downsample.0.weight' \
 'layer2.0.downsample.1.weight' \
 'layer2.0.downsample.1.bias' \
 'layer2.1.conv1.weight' \
 'layer2.1.bn1.weight' \
 'layer2.1.bn1.bias' \
 'layer2.1.conv2.weight' \
 'layer2.1.bn2.weight' \
 'layer2.1.bn2.bias' \
 'layer3.0.conv1.weight' \
 'layer3.0.bn1.weight' \
 'layer3.0.bn1.bias' \
 'layer3.0.conv2.weight' \
 'layer3.0.bn2.weight' \
 'layer3.0.bn2.bias' \
 'layer3.0.downsample.0.weight' \
 'layer3.0.downsample.1.weight' \
 'layer3.0.downsample.1.bias' \
 'layer3.1.conv1.weight' \
 'layer3.1.bn1.weight' \
 'layer3.1.bn1.bias' \
 'layer3.1.conv2.weight' \
 'layer3.1.bn2.weight' \
 'layer3.1.bn2.bias' \
 'layer4.0.conv1.weight' \
 'layer4.0.bn1.weight' \
 'layer4.0.bn1.bias' \
 'layer4.0.conv2.weight' \
 'layer4.0.bn2.weight' \
 'layer4.0.bn2.bias' \
 'layer4.0.downsample.0.weight' \
 'layer4.0.downsample.1.weight' \
 'layer4.0.downsample.1.bias' \
 'layer4.1.conv1.weight' \
 'layer4.1.bn1.weight' \
 'layer4.1.bn1.bias' \
 'layer4.1.conv2.weight' \
 'layer4.1.bn2.weight' \
 'layer4.1.bn2.bias' \
 'fc.weight' \
 'fc.bias'
do
    python3 layerwise_tsne.py --cuda 3 -m ResNet18 -d CIFAR10 -r "model=ResNet18,dataset=CIFAR10" -l $layer_name -p 18
done