#!/bin/bash


#################################################################################
#################################################################################
############################# full g decomp #####################################
#################################################################################
#################################################################################

#################################################################################
# full g decomposition for cifar10+Resnet18
#################################################################################
python3 full_g_decomp.py --cuda 0 -d CIFAR10 -m ResNet18 -r "model=ResNet18,dataset=CIFAR10" 
#################################################################################
# full g decomposition for cifar10+DenseNet3_40
#################################################################################
python3 full_g_decomp.py --cuda 0 -d CIFAR10 -m DenseNet3_40 -r "model=DenseNet3_40,dataset=CIFAR10"
#################################################################################
# full g decomposition for cifar10+vgg11_bn
#################################################################################
python3 full_g_decomp.py --cuda 0 -d CIFAR10 -m VGG11_bn -r "model=VGG11_bn,dataset=CIFAR10"
#################################################################################
# full g decomposition for mnist+vgg11_bn
#################################################################################
python3 full_g_decomp.py --cuda 0 -d MNIST -m VGG11_bn -r "model=VGG11_bn,dataset=MNIST"
#################################################################################
# full g decomposition for mnist+resnet18
#################################################################################
python3 full_g_decomp.py --cuda 1 -d MNIST -m ResNet18 -r "model=ResNet18,dataset=MNIST"
#################################################################################
# full g decomposition for fashionmnist+vgg11_bn
#################################################################################
python3 full_g_decomp.py --cuda 0 -d FashionMNIST -m VGG11_bn -r "model=VGG11_bn,dataset=FashionMNIST"
#################################################################################




















#################################################################################
#################################################################################
####################### layerwise hessian decomposition #########################
#################################################################################
#################################################################################

#################################################################################
# layerwise hessian decomposition for densenet+cifar10
#################################################################################
for layer_name in 'conv1.weight' \
 'block1.layer.0.bn1.weight' \
 'block1.layer.0.bn1.bias' \
 'block1.layer.0.conv1.weight' \
 'block1.layer.1.bn1.weight' \
 'block1.layer.1.bn1.bias' \
 'block1.layer.1.conv1.weight' \
 'block1.layer.2.bn1.weight' \
 'block1.layer.2.bn1.bias' \
 'block1.layer.2.conv1.weight' \
 'block1.layer.3.bn1.weight' \
 'block1.layer.3.bn1.bias' \
 'block1.layer.3.conv1.weight' \
 'block1.layer.4.bn1.weight' \
 'block1.layer.4.bn1.bias' \
 'block1.layer.4.conv1.weight' \
 'block1.layer.5.bn1.weight' \
 'block1.layer.5.bn1.bias' \
 'block1.layer.5.conv1.weight' \
 'block1.layer.6.bn1.weight' \
 'block1.layer.6.bn1.bias' \
 'block1.layer.6.conv1.weight' \
 'block1.layer.7.bn1.weight' \
 'block1.layer.7.bn1.bias' \
 'block1.layer.7.conv1.weight' \
 'block1.layer.8.bn1.weight' \
 'block1.layer.8.bn1.bias' \
 'block1.layer.8.conv1.weight' \
 'block1.layer.9.bn1.weight' \
 'block1.layer.9.bn1.bias' \
 'block1.layer.9.conv1.weight' \
 'block1.layer.10.bn1.weight' \
 'block1.layer.10.bn1.bias' \
 'block1.layer.10.conv1.weight' \
 'block1.layer.11.bn1.weight' \
 'block1.layer.11.bn1.bias' \
 'block1.layer.11.conv1.weight' \
 'trans1.bn1.weight' \
 'trans1.bn1.bias' \
 'trans1.conv1.weight' \
 'block2.layer.0.bn1.weight' \
 'block2.layer.0.bn1.bias' \
 'block2.layer.0.conv1.weight' \
 'block2.layer.1.bn1.weight' \
 'block2.layer.1.bn1.bias' \
 'block2.layer.1.conv1.weight' \
 'block2.layer.2.bn1.weight' \
 'block2.layer.2.bn1.bias' \
 'block2.layer.2.conv1.weight' \
 'block2.layer.3.bn1.weight' \
 'block2.layer.3.bn1.bias' \
 'block2.layer.3.conv1.weight' \
 'block2.layer.4.bn1.weight' \
 'block2.layer.4.bn1.bias' \
 'block2.layer.4.conv1.weight' \
 'block2.layer.5.bn1.weight' \
 'block2.layer.5.bn1.bias' \
 'block2.layer.5.conv1.weight' \
 'block2.layer.6.bn1.weight' \
 'block2.layer.6.bn1.bias' \
 'block2.layer.6.conv1.weight' \
 'block2.layer.7.bn1.weight' \
 'block2.layer.7.bn1.bias' \
 'block2.layer.7.conv1.weight' \
 'block2.layer.8.bn1.weight' \
 'block2.layer.8.bn1.bias' \
 'block2.layer.8.conv1.weight' \
 'block2.layer.9.bn1.weight' \
 'block2.layer.9.bn1.bias' \
 'block2.layer.9.conv1.weight' \
 'block2.layer.10.bn1.weight' \
 'block2.layer.10.bn1.bias' \
 'block2.layer.10.conv1.weight' \
 'block2.layer.11.bn1.weight' \
 'block2.layer.11.bn1.bias' \
 'block2.layer.11.conv1.weight' \
 'trans2.bn1.weight' \
 'trans2.bn1.bias' \
 'trans2.conv1.weight' \
 'block3.layer.0.bn1.weight' \
 'block3.layer.0.bn1.bias' \
 'block3.layer.0.conv1.weight' \
 'block3.layer.1.bn1.weight' \
 'block3.layer.1.bn1.bias' \
 'block3.layer.1.conv1.weight' \
 'block3.layer.2.bn1.weight' \
 'block3.layer.2.bn1.bias' \
 'block3.layer.2.conv1.weight' \
 'block3.layer.3.bn1.weight' \
 'block3.layer.3.bn1.bias' \
 'block3.layer.3.conv1.weight' \
 'block3.layer.4.bn1.weight' \
 'block3.layer.4.bn1.bias' \
 'block3.layer.4.conv1.weight' \
 'block3.layer.5.bn1.weight' \
 'block3.layer.5.bn1.bias' \
 'block3.layer.5.conv1.weight' \
 'block3.layer.6.bn1.weight' \
 'block3.layer.6.bn1.bias' \
 'block3.layer.6.conv1.weight' \
 'block3.layer.7.bn1.weight' \
 'block3.layer.7.bn1.bias' \
 'block3.layer.7.conv1.weight' \
 'block3.layer.8.bn1.weight' \
 'block3.layer.8.bn1.bias' \
 'block3.layer.8.conv1.weight' \
 'block3.layer.9.bn1.weight' \
 'block3.layer.9.bn1.bias' \
 'block3.layer.9.conv1.weight' \
 'block3.layer.10.bn1.weight' \
 'block3.layer.10.bn1.bias' \
 'block3.layer.10.conv1.weight' \
 'block3.layer.11.bn1.weight' \
 'block3.layer.11.bn1.bias' \
 'block3.layer.11.conv1.weight' \
 'bn1.weight' \
 'bn1.bias' \
 'fc.weight' \
 'fc.bias'
do 
    python3 layerwise_hessian_decomp.py --cuda 1 -m DenseNet3_40 -d CIFAR10 -r "model=DenseNet3_40,dataset=CIFAR10" -l $layer_name 
done
#################################################################################
# layerwise hessian decomposition for densenet+fashionmnist
#################################################################################
for layer_name in 'conv1.weight' \
 'block1.layer.0.bn1.weight' \
 'block1.layer.0.bn1.bias' \
 'block1.layer.0.conv1.weight' \
 'block1.layer.1.bn1.weight' \
 'block1.layer.1.bn1.bias' \
 'block1.layer.1.conv1.weight' \
 'block1.layer.2.bn1.weight' \
 'block1.layer.2.bn1.bias' \
 'block1.layer.2.conv1.weight' \
 'block1.layer.3.bn1.weight' \
 'block1.layer.3.bn1.bias' \
 'block1.layer.3.conv1.weight' \
 'block1.layer.4.bn1.weight' \
 'block1.layer.4.bn1.bias' \
 'block1.layer.4.conv1.weight' \
 'block1.layer.5.bn1.weight' \
 'block1.layer.5.bn1.bias' \
 'block1.layer.5.conv1.weight' \
 'block1.layer.6.bn1.weight' \
 'block1.layer.6.bn1.bias' \
 'block1.layer.6.conv1.weight' \
 'block1.layer.7.bn1.weight' \
 'block1.layer.7.bn1.bias' \
 'block1.layer.7.conv1.weight' \
 'block1.layer.8.bn1.weight' \
 'block1.layer.8.bn1.bias' \
 'block1.layer.8.conv1.weight' \
 'block1.layer.9.bn1.weight' \
 'block1.layer.9.bn1.bias' \
 'block1.layer.9.conv1.weight' \
 'block1.layer.10.bn1.weight' \
 'block1.layer.10.bn1.bias' \
 'block1.layer.10.conv1.weight' \
 'block1.layer.11.bn1.weight' \
 'block1.layer.11.bn1.bias' \
 'block1.layer.11.conv1.weight' \
 'trans1.bn1.weight' \
 'trans1.bn1.bias' \
 'trans1.conv1.weight' \
 'block2.layer.0.bn1.weight' \
 'block2.layer.0.bn1.bias' \
 'block2.layer.0.conv1.weight' \
 'block2.layer.1.bn1.weight' \
 'block2.layer.1.bn1.bias' \
 'block2.layer.1.conv1.weight' \
 'block2.layer.2.bn1.weight' \
 'block2.layer.2.bn1.bias' \
 'block2.layer.2.conv1.weight' \
 'block2.layer.3.bn1.weight' \
 'block2.layer.3.bn1.bias' \
 'block2.layer.3.conv1.weight' \
 'block2.layer.4.bn1.weight' \
 'block2.layer.4.bn1.bias' \
 'block2.layer.4.conv1.weight' \
 'block2.layer.5.bn1.weight' \
 'block2.layer.5.bn1.bias' \
 'block2.layer.5.conv1.weight' \
 'block2.layer.6.bn1.weight' \
 'block2.layer.6.bn1.bias' \
 'block2.layer.6.conv1.weight' \
 'block2.layer.7.bn1.weight' \
 'block2.layer.7.bn1.bias' \
 'block2.layer.7.conv1.weight' \
 'block2.layer.8.bn1.weight' \
 'block2.layer.8.bn1.bias' \
 'block2.layer.8.conv1.weight' \
 'block2.layer.9.bn1.weight' \
 'block2.layer.9.bn1.bias' \
 'block2.layer.9.conv1.weight' \
 'block2.layer.10.bn1.weight' \
 'block2.layer.10.bn1.bias' \
 'block2.layer.10.conv1.weight' \
 'block2.layer.11.bn1.weight' \
 'block2.layer.11.bn1.bias' \
 'block2.layer.11.conv1.weight' \
 'trans2.bn1.weight' \
 'trans2.bn1.bias' \
 'trans2.conv1.weight' \
 'block3.layer.0.bn1.weight' \
 'block3.layer.0.bn1.bias' \
 'block3.layer.0.conv1.weight' \
 'block3.layer.1.bn1.weight' \
 'block3.layer.1.bn1.bias' \
 'block3.layer.1.conv1.weight' \
 'block3.layer.2.bn1.weight' \
 'block3.layer.2.bn1.bias' \
 'block3.layer.2.conv1.weight' \
 'block3.layer.3.bn1.weight' \
 'block3.layer.3.bn1.bias' \
 'block3.layer.3.conv1.weight' \
 'block3.layer.4.bn1.weight' \
 'block3.layer.4.bn1.bias' \
 'block3.layer.4.conv1.weight' \
 'block3.layer.5.bn1.weight' \
 'block3.layer.5.bn1.bias' \
 'block3.layer.5.conv1.weight' \
 'block3.layer.6.bn1.weight' \
 'block3.layer.6.bn1.bias' \
 'block3.layer.6.conv1.weight' \
 'block3.layer.7.bn1.weight' \
 'block3.layer.7.bn1.bias' \
 'block3.layer.7.conv1.weight' \
 'block3.layer.8.bn1.weight' \
 'block3.layer.8.bn1.bias' \
 'block3.layer.8.conv1.weight' \
 'block3.layer.9.bn1.weight' \
 'block3.layer.9.bn1.bias' \
 'block3.layer.9.conv1.weight' \
 'block3.layer.10.bn1.weight' \
 'block3.layer.10.bn1.bias' \
 'block3.layer.10.conv1.weight' \
 'block3.layer.11.bn1.weight' \
 'block3.layer.11.bn1.bias' \
 'block3.layer.11.conv1.weight' \
 'bn1.weight' \
 'bn1.bias' \
 'fc.weight' \
 'fc.bias'
do 
    python3 layerwise_hessian_decomp.py --cuda 2 -m DenseNet3_40 -d FashionMNIST -r "model=DenseNet3_40,dataset=FashionMNIST" -l $layer_name 
done
########################################################################################################
# layerwise hessian decomposition for ResNet18+CIFAR10 
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
    python3 layerwise_hessian_decomp.py --cuda 3 -m ResNet18 -d CIFAR10 -r "model=ResNet18,dataset=CIFAR10" -l $layer_name
done
########################################################################################################
# layerwise hessian decomposition for ResNet18+mnist 
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
    python3 layerwise_hessian_decomp.py --cuda 3 -m ResNet18 -d MNIST -r "model=ResNet18,dataset=MNIST" -l $layer_name
done
########################################################################################################
# layerwise g decomposition for vgg11_bn+cifar10 
########################################################################################################
for layer_name in 'features.0.weight' \
 'features.0.bias' \
 'features.1.weight' \
 'features.1.bias' \
 'features.4.weight' \
 'features.4.bias' \
 'features.5.weight' \
 'features.5.bias' \
 'features.8.weight' \
 'features.8.bias' \
 'features.9.weight' \
 'features.9.bias' \
 'features.11.weight' \
 'features.11.bias' \
 'features.12.weight' \
 'features.12.bias' \
 'features.15.weight' \
 'features.15.bias' \
 'features.16.weight' \
 'features.16.bias' \
 'features.18.weight' \
 'features.18.bias' \
 'features.19.weight' \
 'features.19.bias' \
 'features.22.weight' \
 'features.22.bias' \
 'features.23.weight' \
 'features.23.bias' \
 'features.25.weight' \
 'features.25.bias' \
 'features.26.weight' \
 'features.26.bias' \
 'classifier.0.weight' \
 'classifier.1.weight' \
 'classifier.1.bias' \
 'classifier.3.weight' \
 'classifier.4.weight' \
 'classifier.4.bias' \
 'classifier.6.weight' \
 'classifier.6.bias'
do
    python3 layerwise_hessian_decomp.py --cuda 3 -m VGG11_bn -d CIFAR10 -r "model=VGG11_bn,dataset=CIFAR10" -l $layer_name
done
########################################################################################################
# layerwise g decomposition for vgg11_bn+fashionmnist 
########################################################################################################
for layer_name in 'features.0.weight' \
 'features.0.bias' \
 'features.1.weight' \
 'features.1.bias' \
 'features.4.weight' \
 'features.4.bias' \
 'features.5.weight' \
 'features.5.bias' \
 'features.8.weight' \
 'features.8.bias' \
 'features.9.weight' \
 'features.9.bias' \
 'features.11.weight' \
 'features.11.bias' \
 'features.12.weight' \
 'features.12.bias' \
 'features.15.weight' \
 'features.15.bias' \
 'features.16.weight' \
 'features.16.bias' \
 'features.18.weight' \
 'features.18.bias' \
 'features.19.weight' \
 'features.19.bias' \
 'features.22.weight' \
 'features.22.bias' \
 'features.23.weight' \
 'features.23.bias' \
 'features.25.weight' \
 'features.25.bias' \
 'features.26.weight' \
 'features.26.bias' \
 'classifier.0.weight' \
 'classifier.1.weight' \
 'classifier.1.bias' \
 'classifier.3.weight' \
 'classifier.4.weight' \
 'classifier.4.bias' \
 'classifier.6.weight' \
 'classifier.6.bias'
do
    python3 layerwise_hessian_decomp.py --cuda 3 -m VGG11_bn -d FashionMNIST -r "model=VGG11_bn,dataset=FashionMNIST" -l $layer_name
done
########################################################################################################
# layerwise g decomposition for vgg11_bn+mnist 
########################################################################################################
for layer_name in 'features.0.weight' \
 'features.0.bias' \
 'features.1.weight' \
 'features.1.bias' \
 'features.4.weight' \
 'features.4.bias' \
 'features.5.weight' \
 'features.5.bias' \
 'features.8.weight' \
 'features.8.bias' \
 'features.9.weight' \
 'features.9.bias' \
 'features.11.weight' \
 'features.11.bias' \
 'features.12.weight' \
 'features.12.bias' \
 'features.15.weight' \
 'features.15.bias' \
 'features.16.weight' \
 'features.16.bias' \
 'features.18.weight' \
 'features.18.bias' \
 'features.19.weight' \
 'features.19.bias' \
 'features.22.weight' \
 'features.22.bias' \
 'features.23.weight' \
 'features.23.bias' \
 'features.25.weight' \
 'features.25.bias' \
 'features.26.weight' \
 'features.26.bias' \
 'classifier.0.weight' \
 'classifier.1.weight' \
 'classifier.1.bias' \
 'classifier.3.weight' \
 'classifier.4.weight' \
 'classifier.4.bias' \
 'classifier.6.weight' \
 'classifier.6.bias'
do
    python3 layerwise_hessian_decomp.py --cuda 3 -m VGG11_bn -d MNIST -r "model=VGG11_bn,dataset=MNIST" -l $layer_name
done
#################################################################################
# layerwise g decomposition of cifar10+DenseNet3_40
#################################################################################
for layer_name in 'conv1.weight' \
 'block1.layer.0.bn1.weight' \
 'block1.layer.0.bn1.bias' \
 'block1.layer.0.conv1.weight' \
 'block1.layer.1.bn1.weight' \
 'block1.layer.1.bn1.bias' \
 'block1.layer.1.conv1.weight' \
 'block1.layer.2.bn1.weight' \
 'block1.layer.2.bn1.bias' \
 'block1.layer.2.conv1.weight' \
 'block1.layer.3.bn1.weight' \
 'block1.layer.3.bn1.bias' \
 'block1.layer.3.conv1.weight' \
 'block1.layer.4.bn1.weight' \
 'block1.layer.4.bn1.bias' \
 'block1.layer.4.conv1.weight' \
 'block1.layer.5.bn1.weight' \
 'block1.layer.5.bn1.bias' \
 'block1.layer.5.conv1.weight' \
 'block1.layer.6.bn1.weight' \
 'block1.layer.6.bn1.bias' \
 'block1.layer.6.conv1.weight' \
 'block1.layer.7.bn1.weight' \
 'block1.layer.7.bn1.bias' \
 'block1.layer.7.conv1.weight' \
 'block1.layer.8.bn1.weight' \
 'block1.layer.8.bn1.bias' \
 'block1.layer.8.conv1.weight' \
 'block1.layer.9.bn1.weight' \
 'block1.layer.9.bn1.bias' \
 'block1.layer.9.conv1.weight' \
 'block1.layer.10.bn1.weight' \
 'block1.layer.10.bn1.bias' \
 'block1.layer.10.conv1.weight' \
 'block1.layer.11.bn1.weight' \
 'block1.layer.11.bn1.bias' \
 'block1.layer.11.conv1.weight' \
 'trans1.bn1.weight' \
 'trans1.bn1.bias' \
 'trans1.conv1.weight' \
 'block2.layer.0.bn1.weight' \
 'block2.layer.0.bn1.bias' \
 'block2.layer.0.conv1.weight' \
 'block2.layer.1.bn1.weight' \
 'block2.layer.1.bn1.bias' \
 'block2.layer.1.conv1.weight' \
 'block2.layer.2.bn1.weight' \
 'block2.layer.2.bn1.bias' \
 'block2.layer.2.conv1.weight' \
 'block2.layer.3.bn1.weight' \
 'block2.layer.3.bn1.bias' \
 'block2.layer.3.conv1.weight' \
 'block2.layer.4.bn1.weight' \
 'block2.layer.4.bn1.bias' \
 'block2.layer.4.conv1.weight' \
 'block2.layer.5.bn1.weight' \
 'block2.layer.5.bn1.bias' \
 'block2.layer.5.conv1.weight' \
 'block2.layer.6.bn1.weight' \
 'block2.layer.6.bn1.bias' \
 'block2.layer.6.conv1.weight' \
 'block2.layer.7.bn1.weight' \
 'block2.layer.7.bn1.bias' \
 'block2.layer.7.conv1.weight' \
 'block2.layer.8.bn1.weight' \
 'block2.layer.8.bn1.bias' \
 'block2.layer.8.conv1.weight' \
 'block2.layer.9.bn1.weight' \
 'block2.layer.9.bn1.bias' \
 'block2.layer.9.conv1.weight' \
 'block2.layer.10.bn1.weight' \
 'block2.layer.10.bn1.bias' \
 'block2.layer.10.conv1.weight' \
 'block2.layer.11.bn1.weight' \
 'block2.layer.11.bn1.bias' \
 'block2.layer.11.conv1.weight' \
 'trans2.bn1.weight' \
 'trans2.bn1.bias' \
 'trans2.conv1.weight' \
 'block3.layer.0.bn1.weight' \
 'block3.layer.0.bn1.bias' \
 'block3.layer.0.conv1.weight' \
 'block3.layer.1.bn1.weight' \
 'block3.layer.1.bn1.bias' \
 'block3.layer.1.conv1.weight' \
 'block3.layer.2.bn1.weight' \
 'block3.layer.2.bn1.bias' \
 'block3.layer.2.conv1.weight' \
 'block3.layer.3.bn1.weight' \
 'block3.layer.3.bn1.bias' \
 'block3.layer.3.conv1.weight' \
 'block3.layer.4.bn1.weight' \
 'block3.layer.4.bn1.bias' \
 'block3.layer.4.conv1.weight' \
 'block3.layer.5.bn1.weight' \
 'block3.layer.5.bn1.bias' \
 'block3.layer.5.conv1.weight' \
 'block3.layer.6.bn1.weight' \
 'block3.layer.6.bn1.bias' \
 'block3.layer.6.conv1.weight' \
 'block3.layer.7.bn1.weight' \
 'block3.layer.7.bn1.bias' \
 'block3.layer.7.conv1.weight' \
 'block3.layer.8.bn1.weight' \
 'block3.layer.8.bn1.bias' \
 'block3.layer.8.conv1.weight' \
 'block3.layer.9.bn1.weight' \
 'block3.layer.9.bn1.bias' \
 'block3.layer.9.conv1.weight' \
 'block3.layer.10.bn1.weight' \
 'block3.layer.10.bn1.bias' \
 'block3.layer.10.conv1.weight' \
 'block3.layer.11.bn1.weight' \
 'block3.layer.11.bn1.bias' \
 'block3.layer.11.conv1.weight' \
 'bn1.weight' \
 'bn1.bias' \
 'fc.weight' \
 'fc.bias'
do 
    python3 layerwise_g_decomp.py --cuda 0 -m DenseNet3_40 -d CIFAR10 -r "model=DenseNet3_40,dataset=CIFAR10" -l $layer_name 
done
#################################################################################
# layerwise g decomposition of fashionmnist+DenseNet3_40
#################################################################################
for layer_name in 'conv1.weight' \
 'block1.layer.0.bn1.weight' \
 'block1.layer.0.bn1.bias' \
 'block1.layer.0.conv1.weight' \
 'block1.layer.1.bn1.weight' \
 'block1.layer.1.bn1.bias' \
 'block1.layer.1.conv1.weight' \
 'block1.layer.2.bn1.weight' \
 'block1.layer.2.bn1.bias' \
 'block1.layer.2.conv1.weight' \
 'block1.layer.3.bn1.weight' \
 'block1.layer.3.bn1.bias' \
 'block1.layer.3.conv1.weight' \
 'block1.layer.4.bn1.weight' \
 'block1.layer.4.bn1.bias' \
 'block1.layer.4.conv1.weight' \
 'block1.layer.5.bn1.weight' \
 'block1.layer.5.bn1.bias' \
 'block1.layer.5.conv1.weight' \
 'block1.layer.6.bn1.weight' \
 'block1.layer.6.bn1.bias' \
 'block1.layer.6.conv1.weight' \
 'block1.layer.7.bn1.weight' \
 'block1.layer.7.bn1.bias' \
 'block1.layer.7.conv1.weight' \
 'block1.layer.8.bn1.weight' \
 'block1.layer.8.bn1.bias' \
 'block1.layer.8.conv1.weight' \
 'block1.layer.9.bn1.weight' \
 'block1.layer.9.bn1.bias' \
 'block1.layer.9.conv1.weight' \
 'block1.layer.10.bn1.weight' \
 'block1.layer.10.bn1.bias' \
 'block1.layer.10.conv1.weight' \
 'block1.layer.11.bn1.weight' \
 'block1.layer.11.bn1.bias' \
 'block1.layer.11.conv1.weight' \
 'trans1.bn1.weight' \
 'trans1.bn1.bias' \
 'trans1.conv1.weight' \
 'block2.layer.0.bn1.weight' \
 'block2.layer.0.bn1.bias' \
 'block2.layer.0.conv1.weight' \
 'block2.layer.1.bn1.weight' \
 'block2.layer.1.bn1.bias' \
 'block2.layer.1.conv1.weight' \
 'block2.layer.2.bn1.weight' \
 'block2.layer.2.bn1.bias' \
 'block2.layer.2.conv1.weight' \
 'block2.layer.3.bn1.weight' \
 'block2.layer.3.bn1.bias' \
 'block2.layer.3.conv1.weight' \
 'block2.layer.4.bn1.weight' \
 'block2.layer.4.bn1.bias' \
 'block2.layer.4.conv1.weight' \
 'block2.layer.5.bn1.weight' \
 'block2.layer.5.bn1.bias' \
 'block2.layer.5.conv1.weight' \
 'block2.layer.6.bn1.weight' \
 'block2.layer.6.bn1.bias' \
 'block2.layer.6.conv1.weight' \
 'block2.layer.7.bn1.weight' \
 'block2.layer.7.bn1.bias' \
 'block2.layer.7.conv1.weight' \
 'block2.layer.8.bn1.weight' \
 'block2.layer.8.bn1.bias' \
 'block2.layer.8.conv1.weight' \
 'block2.layer.9.bn1.weight' \
 'block2.layer.9.bn1.bias' \
 'block2.layer.9.conv1.weight' \
 'block2.layer.10.bn1.weight' \
 'block2.layer.10.bn1.bias' \
 'block2.layer.10.conv1.weight' \
 'block2.layer.11.bn1.weight' \
 'block2.layer.11.bn1.bias' \
 'block2.layer.11.conv1.weight' \
 'trans2.bn1.weight' \
 'trans2.bn1.bias' \
 'trans2.conv1.weight' \
 'block3.layer.0.bn1.weight' \
 'block3.layer.0.bn1.bias' \
 'block3.layer.0.conv1.weight' \
 'block3.layer.1.bn1.weight' \
 'block3.layer.1.bn1.bias' \
 'block3.layer.1.conv1.weight' \
 'block3.layer.2.bn1.weight' \
 'block3.layer.2.bn1.bias' \
 'block3.layer.2.conv1.weight' \
 'block3.layer.3.bn1.weight' \
 'block3.layer.3.bn1.bias' \
 'block3.layer.3.conv1.weight' \
 'block3.layer.4.bn1.weight' \
 'block3.layer.4.bn1.bias' \
 'block3.layer.4.conv1.weight' \
 'block3.layer.5.bn1.weight' \
 'block3.layer.5.bn1.bias' \
 'block3.layer.5.conv1.weight' \
 'block3.layer.6.bn1.weight' \
 'block3.layer.6.bn1.bias' \
 'block3.layer.6.conv1.weight' \
 'block3.layer.7.bn1.weight' \
 'block3.layer.7.bn1.bias' \
 'block3.layer.7.conv1.weight' \
 'block3.layer.8.bn1.weight' \
 'block3.layer.8.bn1.bias' \
 'block3.layer.8.conv1.weight' \
 'block3.layer.9.bn1.weight' \
 'block3.layer.9.bn1.bias' \
 'block3.layer.9.conv1.weight' \
 'block3.layer.10.bn1.weight' \
 'block3.layer.10.bn1.bias' \
 'block3.layer.10.conv1.weight' \
 'block3.layer.11.bn1.weight' \
 'block3.layer.11.bn1.bias' \
 'block3.layer.11.conv1.weight' \
 'bn1.weight' \
 'bn1.bias' \
 'fc.weight' \
 'fc.bias'
do 
    python3 layerwise_g_decomp.py --cuda 1 -m DenseNet3_40 -d FashionMNIST -r "model=DenseNet3_40,dataset=FashionMNIST" -l $layer_name 
done
########################################################################################################
# layerwise g decomposition for vgg11_bn+fashionmnist 
########################################################################################################
for layer_name in 'features.0.weight' \
 'features.0.bias' \
 'features.1.weight' \
 'features.1.bias' \
 'features.4.weight' \
 'features.4.bias' \
 'features.5.weight' \
 'features.5.bias' \
 'features.8.weight' \
 'features.8.bias' \
 'features.9.weight' \
 'features.9.bias' \
 'features.11.weight' \
 'features.11.bias' \
 'features.12.weight' \
 'features.12.bias' \
 'features.15.weight' \
 'features.15.bias' \
 'features.16.weight' \
 'features.16.bias' \
 'features.18.weight' \
 'features.18.bias' \
 'features.19.weight' \
 'features.19.bias' \
 'features.22.weight' \
 'features.22.bias' \
 'features.23.weight' \
 'features.23.bias' \
 'features.25.weight' \
 'features.25.bias' \
 'features.26.weight' \
 'features.26.bias' \
 'classifier.0.weight' \
 'classifier.1.weight' \
 'classifier.1.bias' \
 'classifier.3.weight' \
 'classifier.4.weight' \
 'classifier.4.bias' \
 'classifier.6.weight' \
 'classifier.6.bias'
do
    python3 layerwise_g_decomp.py --cuda 2 -m VGG11_bn -d FashionMNIST -r "model=VGG11_bn,dataset=FashionMNIST" -l $layer_name
done
########################################################################################################
# layerwise g-decomposition for ResNet18+CIFAR10 
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
    python3 layerwise_g_decomp.py --cuda 0 -m ResNet18 -d CIFAR10 -r "model=ResNet18,dataset=CIFAR10" -l $layer_name
done
########################################################################################################
# layerwise g decomposition for vgg11_bn+cifar10 
########################################################################################################
for layer_name in 'features.0.weight' \
 'features.0.bias' \
 'features.1.weight' \
 'features.1.bias' \
 'features.4.weight' \
 'features.4.bias' \
 'features.5.weight' \
 'features.5.bias' \
 'features.8.weight' \
 'features.8.bias' \
 'features.9.weight' \
 'features.9.bias' \
 'features.11.weight' \
 'features.11.bias' \
 'features.12.weight' \
 'features.12.bias' \
 'features.15.weight' \
 'features.15.bias' \
 'features.16.weight' \
 'features.16.bias' \
 'features.18.weight' \
 'features.18.bias' \
 'features.19.weight' \
 'features.19.bias' \
 'features.22.weight' \
 'features.22.bias' \
 'features.23.weight' \
 'features.23.bias' \
 'features.25.weight' \
 'features.25.bias' \
 'features.26.weight' \
 'features.26.bias' \
 'classifier.0.weight' \
 'classifier.1.weight' \
 'classifier.1.bias' \
 'classifier.3.weight' \
 'classifier.4.weight' \
 'classifier.4.bias' \
 'classifier.6.weight' \
 'classifier.6.bias'
do
    python3 layerwise_g_decomp.py --cuda 3 -m VGG11_bn -d CIFAR10 -r "model=VGG11_bn,dataset=CIFAR10" -l $layer_name
done
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
    python3 layerwise_tsne.py --cuda 0 -m ResNet18 -d CIFAR10 -r "model=ResNet18,dataset=CIFAR10" -l $layer_name
done
########################################################################################################
# layerwise tsne for ResNet18+mnist 
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
    python3 layerwise_tsne.py --cuda 3 -m ResNet18 -d MNIST -r "model=ResNet18,dataset=MNIST" -l $layer_name
done
#################################################################################
# layerwise tsne of cifar10+DenseNet3_40
#################################################################################
for layer_name in 'conv1.weight' \
 'block1.layer.0.bn1.weight' \
 'block1.layer.0.bn1.bias' \
 'block1.layer.0.conv1.weight' \
 'block1.layer.1.bn1.weight' \
 'block1.layer.1.bn1.bias' \
 'block1.layer.1.conv1.weight' \
 'block1.layer.2.bn1.weight' \
 'block1.layer.2.bn1.bias' \
 'block1.layer.2.conv1.weight' \
 'block1.layer.3.bn1.weight' \
 'block1.layer.3.bn1.bias' \
 'block1.layer.3.conv1.weight' \
 'block1.layer.4.bn1.weight' \
 'block1.layer.4.bn1.bias' \
 'block1.layer.4.conv1.weight' \
 'block1.layer.5.bn1.weight' \
 'block1.layer.5.bn1.bias' \
 'block1.layer.5.conv1.weight' \
 'block1.layer.6.bn1.weight' \
 'block1.layer.6.bn1.bias' \
 'block1.layer.6.conv1.weight' \
 'block1.layer.7.bn1.weight' \
 'block1.layer.7.bn1.bias' \
 'block1.layer.7.conv1.weight' \
 'block1.layer.8.bn1.weight' \
 'block1.layer.8.bn1.bias' \
 'block1.layer.8.conv1.weight' \
 'block1.layer.9.bn1.weight' \
 'block1.layer.9.bn1.bias' \
 'block1.layer.9.conv1.weight' \
 'block1.layer.10.bn1.weight' \
 'block1.layer.10.bn1.bias' \
 'block1.layer.10.conv1.weight' \
 'block1.layer.11.bn1.weight' \
 'block1.layer.11.bn1.bias' \
 'block1.layer.11.conv1.weight' \
 'trans1.bn1.weight' \
 'trans1.bn1.bias' \
 'trans1.conv1.weight' \
 'block2.layer.0.bn1.weight' \
 'block2.layer.0.bn1.bias' \
 'block2.layer.0.conv1.weight' \
 'block2.layer.1.bn1.weight' \
 'block2.layer.1.bn1.bias' \
 'block2.layer.1.conv1.weight' \
 'block2.layer.2.bn1.weight' \
 'block2.layer.2.bn1.bias' \
 'block2.layer.2.conv1.weight' \
 'block2.layer.3.bn1.weight' \
 'block2.layer.3.bn1.bias' \
 'block2.layer.3.conv1.weight' \
 'block2.layer.4.bn1.weight' \
 'block2.layer.4.bn1.bias' \
 'block2.layer.4.conv1.weight' \
 'block2.layer.5.bn1.weight' \
 'block2.layer.5.bn1.bias' \
 'block2.layer.5.conv1.weight' \
 'block2.layer.6.bn1.weight' \
 'block2.layer.6.bn1.bias' \
 'block2.layer.6.conv1.weight' \
 'block2.layer.7.bn1.weight' \
 'block2.layer.7.bn1.bias' \
 'block2.layer.7.conv1.weight' \
 'block2.layer.8.bn1.weight' \
 'block2.layer.8.bn1.bias' \
 'block2.layer.8.conv1.weight' \
 'block2.layer.9.bn1.weight' \
 'block2.layer.9.bn1.bias' \
 'block2.layer.9.conv1.weight' \
 'block2.layer.10.bn1.weight' \
 'block2.layer.10.bn1.bias' \
 'block2.layer.10.conv1.weight' \
 'block2.layer.11.bn1.weight' \
 'block2.layer.11.bn1.bias' \
 'block2.layer.11.conv1.weight' \
 'trans2.bn1.weight' \
 'trans2.bn1.bias' \
 'trans2.conv1.weight' \
 'block3.layer.0.bn1.weight' \
 'block3.layer.0.bn1.bias' \
 'block3.layer.0.conv1.weight' \
 'block3.layer.1.bn1.weight' \
 'block3.layer.1.bn1.bias' \
 'block3.layer.1.conv1.weight' \
 'block3.layer.2.bn1.weight' \
 'block3.layer.2.bn1.bias' \
 'block3.layer.2.conv1.weight' \
 'block3.layer.3.bn1.weight' \
 'block3.layer.3.bn1.bias' \
 'block3.layer.3.conv1.weight' \
 'block3.layer.4.bn1.weight' \
 'block3.layer.4.bn1.bias' \
 'block3.layer.4.conv1.weight' \
 'block3.layer.5.bn1.weight' \
 'block3.layer.5.bn1.bias' \
 'block3.layer.5.conv1.weight' \
 'block3.layer.6.bn1.weight' \
 'block3.layer.6.bn1.bias' \
 'block3.layer.6.conv1.weight' \
 'block3.layer.7.bn1.weight' \
 'block3.layer.7.bn1.bias' \
 'block3.layer.7.conv1.weight' \
 'block3.layer.8.bn1.weight' \
 'block3.layer.8.bn1.bias' \
 'block3.layer.8.conv1.weight' \
 'block3.layer.9.bn1.weight' \
 'block3.layer.9.bn1.bias' \
 'block3.layer.9.conv1.weight' \
 'block3.layer.10.bn1.weight' \
 'block3.layer.10.bn1.bias' \
 'block3.layer.10.conv1.weight' \
 'block3.layer.11.bn1.weight' \
 'block3.layer.11.bn1.bias' \
 'block3.layer.11.conv1.weight' \
 'bn1.weight' \
 'bn1.bias' \
 'fc.weight' \
 'fc.bias'
do 
    python3 layerwise_tsne.py --cuda 0 -m DenseNet3_40 -d CIFAR10 -r "model=DenseNet3_40,dataset=CIFAR10" -l $layer_name 
done
#################################################################################
# layerwise tsne of fashionmnist+DenseNet3_40
#################################################################################
for layer_name in 'conv1.weight' \
 'block1.layer.0.bn1.weight' \
 'block1.layer.0.bn1.bias' \
 'block1.layer.0.conv1.weight' \
 'block1.layer.1.bn1.weight' \
 'block1.layer.1.bn1.bias' \
 'block1.layer.1.conv1.weight' \
 'block1.layer.2.bn1.weight' \
 'block1.layer.2.bn1.bias' \
 'block1.layer.2.conv1.weight' \
 'block1.layer.3.bn1.weight' \
 'block1.layer.3.bn1.bias' \
 'block1.layer.3.conv1.weight' \
 'block1.layer.4.bn1.weight' \
 'block1.layer.4.bn1.bias' \
 'block1.layer.4.conv1.weight' \
 'block1.layer.5.bn1.weight' \
 'block1.layer.5.bn1.bias' \
 'block1.layer.5.conv1.weight' \
 'block1.layer.6.bn1.weight' \
 'block1.layer.6.bn1.bias' \
 'block1.layer.6.conv1.weight' \
 'block1.layer.7.bn1.weight' \
 'block1.layer.7.bn1.bias' \
 'block1.layer.7.conv1.weight' \
 'block1.layer.8.bn1.weight' \
 'block1.layer.8.bn1.bias' \
 'block1.layer.8.conv1.weight' \
 'block1.layer.9.bn1.weight' \
 'block1.layer.9.bn1.bias' \
 'block1.layer.9.conv1.weight' \
 'block1.layer.10.bn1.weight' \
 'block1.layer.10.bn1.bias' \
 'block1.layer.10.conv1.weight' \
 'block1.layer.11.bn1.weight' \
 'block1.layer.11.bn1.bias' \
 'block1.layer.11.conv1.weight' \
 'trans1.bn1.weight' \
 'trans1.bn1.bias' \
 'trans1.conv1.weight' \
 'block2.layer.0.bn1.weight' \
 'block2.layer.0.bn1.bias' \
 'block2.layer.0.conv1.weight' \
 'block2.layer.1.bn1.weight' \
 'block2.layer.1.bn1.bias' \
 'block2.layer.1.conv1.weight' \
 'block2.layer.2.bn1.weight' \
 'block2.layer.2.bn1.bias' \
 'block2.layer.2.conv1.weight' \
 'block2.layer.3.bn1.weight' \
 'block2.layer.3.bn1.bias' \
 'block2.layer.3.conv1.weight' \
 'block2.layer.4.bn1.weight' \
 'block2.layer.4.bn1.bias' \
 'block2.layer.4.conv1.weight' \
 'block2.layer.5.bn1.weight' \
 'block2.layer.5.bn1.bias' \
 'block2.layer.5.conv1.weight' \
 'block2.layer.6.bn1.weight' \
 'block2.layer.6.bn1.bias' \
 'block2.layer.6.conv1.weight' \
 'block2.layer.7.bn1.weight' \
 'block2.layer.7.bn1.bias' \
 'block2.layer.7.conv1.weight' \
 'block2.layer.8.bn1.weight' \
 'block2.layer.8.bn1.bias' \
 'block2.layer.8.conv1.weight' \
 'block2.layer.9.bn1.weight' \
 'block2.layer.9.bn1.bias' \
 'block2.layer.9.conv1.weight' \
 'block2.layer.10.bn1.weight' \
 'block2.layer.10.bn1.bias' \
 'block2.layer.10.conv1.weight' \
 'block2.layer.11.bn1.weight' \
 'block2.layer.11.bn1.bias' \
 'block2.layer.11.conv1.weight' \
 'trans2.bn1.weight' \
 'trans2.bn1.bias' \
 'trans2.conv1.weight' \
 'block3.layer.0.bn1.weight' \
 'block3.layer.0.bn1.bias' \
 'block3.layer.0.conv1.weight' \
 'block3.layer.1.bn1.weight' \
 'block3.layer.1.bn1.bias' \
 'block3.layer.1.conv1.weight' \
 'block3.layer.2.bn1.weight' \
 'block3.layer.2.bn1.bias' \
 'block3.layer.2.conv1.weight' \
 'block3.layer.3.bn1.weight' \
 'block3.layer.3.bn1.bias' \
 'block3.layer.3.conv1.weight' \
 'block3.layer.4.bn1.weight' \
 'block3.layer.4.bn1.bias' \
 'block3.layer.4.conv1.weight' \
 'block3.layer.5.bn1.weight' \
 'block3.layer.5.bn1.bias' \
 'block3.layer.5.conv1.weight' \
 'block3.layer.6.bn1.weight' \
 'block3.layer.6.bn1.bias' \
 'block3.layer.6.conv1.weight' \
 'block3.layer.7.bn1.weight' \
 'block3.layer.7.bn1.bias' \
 'block3.layer.7.conv1.weight' \
 'block3.layer.8.bn1.weight' \
 'block3.layer.8.bn1.bias' \
 'block3.layer.8.conv1.weight' \
 'block3.layer.9.bn1.weight' \
 'block3.layer.9.bn1.bias' \
 'block3.layer.9.conv1.weight' \
 'block3.layer.10.bn1.weight' \
 'block3.layer.10.bn1.bias' \
 'block3.layer.10.conv1.weight' \
 'block3.layer.11.bn1.weight' \
 'block3.layer.11.bn1.bias' \
 'block3.layer.11.conv1.weight' \
 'bn1.weight' \
 'bn1.bias' \
 'fc.weight' \
 'fc.bias'
do 
    python3 layerwise_tsne.py --cuda 3 -m DenseNet3_40 -d FashionMNIST -r "model=DenseNet3_40,dataset=FashionMNIST" -l $layer_name 
done
########################################################################################################
# layerwise tsne for vgg11_bn+mnist 
########################################################################################################
for layer_name in 'features.0.weight' \
 'features.0.bias' \
 'features.1.weight' \
 'features.1.bias' \
 'features.4.weight' \
 'features.4.bias' \
 'features.5.weight' \
 'features.5.bias' \
 'features.8.weight' \
 'features.8.bias' \
 'features.9.weight' \
 'features.9.bias' \
 'features.11.weight' \
 'features.11.bias' \
 'features.12.weight' \
 'features.12.bias' \
 'features.15.weight' \
 'features.15.bias' \
 'features.16.weight' \
 'features.16.bias' \
 'features.18.weight' \
 'features.18.bias' \
 'features.19.weight' \
 'features.19.bias' \
 'features.22.weight' \
 'features.22.bias' \
 'features.23.weight' \
 'features.23.bias' \
 'features.25.weight' \
 'features.25.bias' \
 'features.26.weight' \
 'features.26.bias' \
 'classifier.0.weight' \
 'classifier.1.weight' \
 'classifier.1.bias' \
 'classifier.3.weight' \
 'classifier.4.weight' \
 'classifier.4.bias' \
 'classifier.6.weight' \
 'classifier.6.bias'
do
    python3 layerwise_tsne.py --cuda 0 -m VGG11_bn -d MNIST -r "model=VGG11_bn,dataset=MNIST" -l $layer_name
done
########################################################################################################
# layerwise tsne for vgg11_bn+cifar10 
########################################################################################################
for layer_name in 'features.0.weight' \
 'features.0.bias' \
 'features.1.weight' \
 'features.1.bias' \
 'features.4.weight' \
 'features.4.bias' \
 'features.5.weight' \
 'features.5.bias' \
 'features.8.weight' \
 'features.8.bias' \
 'features.9.weight' \
 'features.9.bias' \
 'features.11.weight' \
 'features.11.bias' \
 'features.12.weight' \
 'features.12.bias' \
 'features.15.weight' \
 'features.15.bias' \
 'features.16.weight' \
 'features.16.bias' \
 'features.18.weight' \
 'features.18.bias' \
 'features.19.weight' \
 'features.19.bias' \
 'features.22.weight' \
 'features.22.bias' \
 'features.23.weight' \
 'features.23.bias' \
 'features.25.weight' \
 'features.25.bias' \
 'features.26.weight' \
 'features.26.bias' \
 'classifier.0.weight' \
 'classifier.1.weight' \
 'classifier.1.bias' \
 'classifier.3.weight' \
 'classifier.4.weight' \
 'classifier.4.bias' \
 'classifier.6.weight' \
 'classifier.6.bias'
do
    python3 layerwise_tsne.py --cuda 2 -m VGG11_bn -d CIFAR10 -r "model=VGG11_bn,dataset=CIFAR10" -l $layer_name
done
########################################################################################################
# layerwise tsne for vgg11_bn+fashionmnist 
########################################################################################################
for layer_name in 'features.0.weight' \
 'features.0.bias' \
 'features.1.weight' \
 'features.1.bias' \
 'features.4.weight' \
 'features.4.bias' \
 'features.5.weight' \
 'features.5.bias' \
 'features.8.weight' \
 'features.8.bias' \
 'features.9.weight' \
 'features.9.bias' \
 'features.11.weight' \
 'features.11.bias' \
 'features.12.weight' \
 'features.12.bias' \
 'features.15.weight' \
 'features.15.bias' \
 'features.16.weight' \
 'features.16.bias' \
 'features.18.weight' \
 'features.18.bias' \
 'features.19.weight' \
 'features.19.bias' \
 'features.22.weight' \
 'features.22.bias' \
 'features.23.weight' \
 'features.23.bias' \
 'features.25.weight' \
 'features.25.bias' \
 'features.26.weight' \
 'features.26.bias' \
 'classifier.0.weight' \
 'classifier.1.weight' \
 'classifier.1.bias' \
 'classifier.3.weight' \
 'classifier.4.weight' \
 'classifier.4.bias' \
 'classifier.6.weight' \
 'classifier.6.bias'
do
    python3 layerwise_tsne.py --cuda 0 -m VGG11_bn -d FashionMNIST -r "model=VGG11_bn,dataset=FashionMNIST" -l $layer_name
done
########################################################################################################
# layerwise tsne for vgg11_bn+cifar10 
########################################################################################################
for layer_name in 'features.0.weight' \
 'features.0.bias' \
 'features.1.weight' \
 'features.1.bias' \
 'features.4.weight' \
 'features.4.bias' \
 'features.5.weight' \
 'features.5.bias' \
 'features.8.weight' \
 'features.8.bias' \
 'features.9.weight' \
 'features.9.bias' \
 'features.11.weight' \
 'features.11.bias' \
 'features.12.weight' \
 'features.12.bias' \
 'features.15.weight' \
 'features.15.bias' \
 'features.16.weight' \
 'features.16.bias' \
 'features.18.weight' \
 'features.18.bias' \
 'features.19.weight' \
 'features.19.bias' \
 'features.22.weight' \
 'features.22.bias' \
 'features.23.weight' \
 'features.23.bias' \
 'features.25.weight' \
 'features.25.bias' \
 'features.26.weight' \
 'features.26.bias' \
 'classifier.0.weight' \
 'classifier.1.weight' \
 'classifier.1.bias' \
 'classifier.3.weight' \
 'classifier.4.weight' \
 'classifier.4.bias' \
 'classifier.6.weight' \
 'classifier.6.bias'
do
    python3 layerwise_tsne.py --cuda 0 -m VGG11_bn -d CIFAR10 -r "model=VGG11_bn,dataset=CIFAR10" -l $layer_name
done
####################################################################################
# layerwise g decomposition for vgg11_bn+mnist
####################################################################################
for layer_name in 'features.0.weight' \
 'features.0.bias' \
 'features.1.weight' \
 'features.1.bias' \
 'features.4.weight' \
 'features.4.bias' \
 'features.5.weight' \
 'features.5.bias' \
 'features.8.weight' \
 'features.8.bias' \
 'features.9.weight' \
 'features.9.bias' \
 'features.11.weight' \
 'features.11.bias' \
 'features.12.weight' \
 'features.12.bias' \
 'features.15.weight' \
 'features.15.bias' \
 'features.16.weight' \
 'features.16.bias' \
 'features.18.weight' \
 'features.18.bias' \
 'features.19.weight' \
 'features.19.bias' \
 'features.22.weight' \
 'features.22.bias' \
 'features.23.weight' \
 'features.23.bias' \
 'features.25.weight' \
 'features.25.bias' \
 'features.26.weight' \
 'features.26.bias' \
 'classifier.0.weight' \
 'classifier.1.weight' \
 'classifier.1.bias' \
 'classifier.3.weight' \
 'classifier.4.weight' \
 'classifier.4.bias' \
 'classifier.6.weight' \
 'classifier.6.bias'
do 
    python3 layerwise_g_decomp.py --cuda 1 -d MNIST -m VGG11_bn -l $layer_name -r "model=VGG11_bn,dataset=MNIST"
done
####################################################################################
# layerwise g decomposition for vgg11_bn+fashionmnist
####################################################################################
for layer_name in 'features.0.weight' \
 'features.0.bias' \
 'features.1.weight' \
 'features.1.bias' \
 'features.4.weight' \
 'features.4.bias' \
 'features.5.weight' \
 'features.5.bias' \
 'features.8.weight' \
 'features.8.bias' \
 'features.9.weight' \
 'features.9.bias' \
 'features.11.weight' \
 'features.11.bias' \
 'features.12.weight' \
 'features.12.bias' \
 'features.15.weight' \
 'features.15.bias' \
 'features.16.weight' \
 'features.16.bias' \
 'features.18.weight' \
 'features.18.bias' \
 'features.19.weight' \
 'features.19.bias' \
 'features.22.weight' \
 'features.22.bias' \
 'features.23.weight' \
 'features.23.bias' \
 'features.25.weight' \
 'features.25.bias' \
 'features.26.weight' \
 'features.26.bias' \
 'classifier.0.weight' \
 'classifier.1.weight' \
 'classifier.1.bias' \
 'classifier.3.weight' \
 'classifier.4.weight' \
 'classifier.4.bias' \
 'classifier.6.weight' \
 'classifier.6.bias'
do 
    python3 layerwise_g_decomp.py --cuda 1 -d FashionMNIST -m VGG11_bn -l $layer_name -r "model=VGG11_bn,dataset=FashionMNIST"
done
########################################################################################################
# layerwise g decomposition for ResNet18+CIFAR10 
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
    python3 layerwise_g_decomp.py --cuda 2 -m ResNet18 -d MNIST -r "model=ResNet18,dataset=MNIST" -l $layer_name
done
####################################################################################
# full hessian decomposition for DenseNet3_40+CIFAR10
####################################################################################
python3 full_hessian_decomp.py --cuda 2 -m DenseNet3_40 -d CIFAR10 -r "model=DenseNet3_40,dataset=CIFAR10" 
####################################################################################
# full hessian decomposition for ResNet18+CIFAR10
####################################################################################
python3 full_hessian_decomp.py --cuda 2 -m ResNet18 -d CIFAR10 -r "model=ResNet18,dataset=CIFAR10" 
####################################################################################
# full hessian decomposition for VGG11_bn+cifar10 
####################################################################################
python3 full_hessian_decomp.py --cuda 1 -m VGG11_bn -d CIFAR10 -r "model=VGG11_bn,dataset=CIFAR10"
####################################################################################
# full hessian decomposition for cifar100+resnet18
####################################################################################
python3 full_hessian_decomp.py --cuda 1 -m ResNet18 -d CIFAR100 -r "model=ResNet18,dataset=CIFAR100" --para
####################################################################################
# full hessian decomposition for mnist+vgg11bn
####################################################################################
python3 full_hessian_decomp.py --cuda 2 -m VGG11_bn -d MNIST -r "model=VGG11_bn,dataset=MNIST"
####################################################################################
# full hessian decomposition for fashionmnist+vgg11_bn
####################################################################################
python3 full_hessian_decomp.py --cuda 2 -m VGG11_bn -d FashionMNIST -r "model=VGG11_bn,dataset=FashionMNIST"
####################################################################################
# full hessian decompositoin for mnist+resnet18
####################################################################################
python3 full_hessian_decomp.py --cuda 1 -m ResNet18 -d MNIST -r "model=ResNet18,dataset=MNIST"
####################################################################################
# full tsne for ResNet18+CIFAR10
####################################################################################
python3 full_tsne.py --cuda 1 -m ResNet18 -d CIFAR10 -r "model=ResNet18,dataset=CIFAR10" 
####################################################################################
# full tsne for DenseNet3_40+CIFAR10
####################################################################################
python3 full_tsne.py --cuda 1 -m DenseNet3_40 -d CIFAR10 -r "model=DenseNet3_40,dataset=CIFAR10"
####################################################################################
# full tsne for VGG11_bn+CIFAR10 
####################################################################################
python3 full_tsne.py --cuda 1 -m VGG11_bn -d CIFAR10 -r "model=VGG11_bn,dataset=CIFAR10"
####################################################################################
# full tsne for VGG11_bn+MNIST
####################################################################################
python3 full_tsne.py --cuda 1 -m VGG11_bn -d MNIST -r "model=VGG11_bn,dataset=MNIST"
####################################################################################
# full tsne for VGG11_bn+Fashiomnist
####################################################################################
python3 full_tsne.py --cuda 2 -m VGG11_bn -d FashionMNIST -r "model=VGG11_bn,dataset=FashionMNIST"
####################################################################################
# full tsne for resnet18+mnist
####################################################################################
python3 full_tsne.py --cuda 2 -m ResNet18 -d MNIST -r "model=ResNet18,dataset=MNIST"
####################################################################################
# combine all plots
####################################################################################
python combine_plots.py -r "model=DenseNet3_40,dataset=CIFAR10"      -m DenseNet3_40
python combine_plots.py -r "model=DenseNet3_40,dataset=FashionMNIST" -m DenseNet3_40
python combine_plots.py -r "model=ResNet18,dataset=CIFAR10"          -m ResNet18
python combine_plots.py -r "model=ResNet18,dataset=MNIST"            -m ResNet18
python combine_plots.py -r "model=VGG11_bn,dataset=CIFAR10"          -m VGG11_bn
python combine_plots.py -r "model=VGG11_bn,dataset=FashionMNIST"     -m VGG11_bn
python combine_plots.py -r "model=VGG11_bn,dataset=MNIST"            -m VGG11_bn
