#! /usr/bin/python
# -*- coding: utf-8 -*-

import torch as th
import torchvision as tv
import torchvision.transforms as transforms

__author__ = 'fyabc'


def load_dataset():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = tv.datasets.CIFAR10(root='G:/Data/MSRA/CIFAR-10', train=True,
                                   download=True, transform=transform)
    trainloader = th.utils.data.DataLoader(trainset, batch_size=4,
                                           shuffle=True, num_workers=2)

    testset = tv.datasets.CIFAR10(root='G:/Data/MSRA/CIFAR-10', train=False,
                                  download=True, transform=transform)
    testloader = th.utils.data.DataLoader(testset, batch_size=4,
                                          shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, classes


def main():
    load_dataset()


if __name__ == '__main__':
    main()
