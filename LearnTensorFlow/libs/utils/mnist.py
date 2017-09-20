#! /usr/bin/python
# -*- coding: utf-8 -*-

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as _tf

from .constants import floatX

__author__ = 'fyabc'


# Constants.
ImageSize = 28 * 28
LabelSize = 10
TrainIteration = 1000
BatchSize = 100
LearningRate = 0.5


def load_data():
    """
    Get MNIST dataset.
    [NOTE] MNIST dataset attribute:
    mnist.train: 55000
    mnist.validation: 5000
    mnist.test: 10000

    mnist.train.images: (55000, 784) of float
    mnist.train.labels: (55000, 10) of float
    """
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
    return mnist


def input_placeholders():
    # Input placeholder, None in shape means any size.
    x = _tf.placeholder(floatX, [None, ImageSize], name='x')
    # Get ground truth labels.
    y_ = _tf.placeholder(floatX, [None, LabelSize], name='y')

    return x, y_
