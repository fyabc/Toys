#! /usr/bin/python
# -*- coding: utf-8 -*-

import time

import torch as th
import torchvision as tv
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

__author__ = 'fyabc'

TestBatchSize = 4
UseCuda = False


def load_dataset():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = tv.datasets.CIFAR10(root='G:/Data/MSRA/CIFAR-10', train=True,
                                   download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

    testset = tv.datasets.CIFAR10(root='G:/Data/MSRA/CIFAR-10', train=False,
                                  download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=TestBatchSize, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, classes


class Net(nn.Module):
    Channel1 = 64
    Kernel1 = 5
    Channel2 = 16
    Kernel2 = 5

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, self.Channel1, self.Kernel1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(self.Channel1, self.Channel2, self.Kernel2)
        self.fc1 = nn.Linear(self.Channel2 * self.Kernel2 * self.Kernel2, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_test_accuracy(testloader, net, classes):
    class_correct = [0. for _ in range(10)]
    class_total = [0 for _ in range(10)]

    for data in testloader:
        images, labels = data
        if UseCuda:
            images, labels = images.cuda(), labels.cuda()

        outputs = net(Variable(images))
        _, predicted = th.max(outputs.data, 1)
        c = (predicted == labels).squeeze()
        for i in range(TestBatchSize):
            label = labels[i]
            class_correct[label] += c[i]
            class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * sum(class_correct) / sum(class_total)))


def main():
    trainloader, testloader, classes = load_dataset()

    net = Net()
    if UseCuda:
        net.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    time_start = time.time()

    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            if UseCuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f, time: %.3fs' %
                      (epoch + 1, i + 1, running_loss / 2000, time.time() - time_start))
                running_loss = 0.0

    print('Finished Training')

    get_test_accuracy(testloader, net, classes)


if __name__ == '__main__':
    main()
