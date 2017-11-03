#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from ...utils.constants import floatX

__author__ = 'fyabc'


def print_tensor(t):
    print('Tensor:', t.op.name, t.get_shape().as_list())


def inference(images, options):
    """

    :param images: Tensor of images
    :param options: dict for options
    :return: list
        model output, list of parameters
    """

    parameters = []

    # Convolutional 1
    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64], dtype=floatX, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(images, kernel, strides=[1, 4, 4, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=floatX), trainable=True, name='biases')
        biased = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(biased, name=scope)

        parameters += [kernel, biases]
        print_tensor(conv1)

    # Pool 1
    if options.get('lrn', False):
        conv1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9, beta=0.75, name='lrn1')
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')
    print_tensor(pool1)

    # Convolution 2
    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192], dtype=floatX, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool1, kernel, strides=[1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=floatX), trainable=True, name='biases')
        biased = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(biased, name=scope)

        parameters += [kernel, biases]
        print_tensor(conv2)

    # Pool 2
    if options.get('lrn', False):
        conv2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9, beta=0.75, name='lrn1')
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')
    print_tensor(pool2)

    # Convolution 3
    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384], dtype=floatX, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool2, kernel, strides=[1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=floatX), trainable=True, name='biases')
        biased = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(biased, name=scope)

        parameters += [kernel, biases]
        print_tensor(conv3)

    # Convolution 4
    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256], dtype=floatX, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv3, kernel, strides=[1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=floatX), trainable=True, name='biases')
        biased = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(biased, name=scope)

        parameters += [kernel, biases]
        print_tensor(conv4)

    # Convolution 5
    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=floatX, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv4, kernel, strides=[1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=floatX), trainable=True, name='biases')
        biased = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(biased, name=scope)

        parameters += [kernel, biases]
        print_tensor(conv5)

    # Pool 5
    pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool5')
    print_tensor(pool5)

    # FC 1
    with tf.name_scope('fc1') as scope:
        W = tf.Variable(tf.truncated_normal([6 * 6 * 256, 4096], dtype=floatX, stddev=1e-1), name='weights')
        b = tf.Variable(tf.constant(0.0, shape=[4096], dtype=floatX), trainable=True, name='biases')
        fc1 = tf.nn.relu(tf.matmul(tf.reshape(pool5, [-1, 6 * 6 * 256]), W) + b, name=scope)

        parameters += [W, b]
        print_tensor(fc1)

    # FC 2
    with tf.name_scope('fc2') as scope:
        W = tf.Variable(tf.truncated_normal([4096, 4096], dtype=floatX, stddev=1e-1), name='weights')
        b = tf.Variable(tf.constant(0.0, shape=[4096], dtype=floatX), trainable=True, name='biases')
        fc2 = tf.nn.relu(tf.matmul(fc1, W) + b, name=scope)

        parameters += [W, b]
        print_tensor(fc2)

    # FC 3
    with tf.name_scope('fc3') as scope:
        W = tf.Variable(tf.truncated_normal([4096, 1000], dtype=floatX, stddev=1e-1), name='weights')
        b = tf.Variable(tf.constant(0.0, shape=[1000], dtype=floatX), trainable=True, name='biases')
        fc3 = tf.nn.relu(tf.matmul(fc2, W) + b, name=scope)

        parameters += [W, b]
        print_tensor(fc3)

    return fc3, parameters
