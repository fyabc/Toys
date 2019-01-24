#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Tutorial of neural network.

Input is a MNIST image.
"""

import torch as th
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

__author__ = 'fyabc'


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # 1 input image channel, 6 output channels, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # An affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def main():
    net = Net()

    print(net)
    print()

    params = list(net.parameters())
    print(len(params))
    print(params[0].size())
    print()

    # Batch size = 1
    input_ = Variable(th.randn(1, 1, 32, 32))
    out = net(input_)
    print(out)
    print()

    # Zero the gradient buffers of all parameters and backprops with random gradients.
    net.zero_grad()
    net(input_).backward(th.randn(1, 10))

    target = Variable(th.arange(1, 11))  # a dummy target, for example
    target = target.view(1, -1)  # make it the same shape as output
    criterion = nn.MSELoss()

    loss = criterion(out, target)
    print(loss)
    print()

    print(loss.grad_fn)  # MSELoss
    print(loss.grad_fn.next_functions[0][0])  # Linear
    print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU
    print()

    net.zero_grad()  # zeroes the gradient buffers of all parameters
    print('conv1.bias.grad before backward')
    print(net.conv1.bias.grad)
    loss.backward()
    print('conv1.bias.grad after backward')
    print(net.conv1.bias.grad)
    print()

    # Create your optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    # In your training loop:
    optimizer.zero_grad()  # zero the gradient buffers
    output = net(input_)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()  # Does the update


if __name__ == '__main__':
    main()
