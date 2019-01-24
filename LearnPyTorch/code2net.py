#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Try to build network from code (Neural Architecture Search results)."""

import torch as th
import torch.nn as nn

__author__ = 'fyabc'


class NetCode:
    # Layer types.
    Recurrent = 0
    Convolutional = 1
    Attention = 2

    # Recurrent hyperparameters.

    # Convolutional hyperparameters.

    # Attention hyperparameters.


class ChildNet(nn.Module):
    def __init__(self, net_code):
        super().__init__()
        self._net = [self._code2layer(layer_code) for layer_code in net_code]

    def forward(self, x):
        for layer in self._net:
            x = layer(x)
        return x

    def _code2layer(self, layer_code):
        layer_type = layer_code[0]

        if layer_type == NetCode.Recurrent:
            pass
        elif layer_type == NetCode.Convolutional:
            pass
        elif layer_type == NetCode.Attention:
            raise NotImplementedError('Attention layer not implemented')
        else:
            raise ValueError('Unknown layer type {}'.format(layer_type))
