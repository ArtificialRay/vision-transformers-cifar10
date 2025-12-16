#!/bin/bash

# 按顺序执行训练命令
#python train_cifar10.py --dataset cifar100 --net vit --data-path /root/autodl-fs/CIFAR100
python train_cifar10.py --dataset cifar100 --net dyt --data-path /root/autodl-fs/CIFAR100
python train_cifar10.py --dataset cifar100 --net mlpmixer --data-path /root/autodl-fs/CIFAR100
python train_cifar10.py --dataset cifar100 --net cait --data-path /root/autodl-fs/CIFAR100
python train_cifar10.py --dataset cifar100 --net vgg16 --data-path /root/autodl-fs/CIFAR100
python train_cifar10.py --dataset cifar100 --net vgg19 --data-path /root/autodl-fs/CIFAR100
python train_cifar10.py --dataset cifar100 --net res34 --data-path /root/autodl-fs/CIFAR100
#python train_cifar10.py --dataset cifar100 --net res18 --data-path /root/autodl-fs/CIFAR100
# 所有训练完成后关机
shutdown