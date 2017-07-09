#! /usr/bin/python
# -*- coding: utf-8 -*-

import time

import tensorflow as tf

from libs.utils.mnist import *

__author__ = 'fyabc'


# Constants.
DisplayFreq = 100
TrainIteration = 20000
BatchSize = 50


# Initializers.

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_variable(shape):
    # [NOTE] Use a slightly positive initial bias to avoid "dead neurons".
    return tf.Variable(tf.constant(0.1, shape=shape))


# Convolutional layers.

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def main():
    # Get MNIST dataset.
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

    x, y_ = input_placeholders()

    # First convolutional layer

    # Weight shape: (patch_width, patch_height, input_channels, output_channels)
    # Bias shape: (output_channels,)
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    # x_image shape: (data_case, image_width, image_height, color_channels)
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional layer

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # Densely connected layer
    # Reduced image size: 7 * 7, hidden size: 1024
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout
    keep_prob = tf.placeholder(floatX)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Readout layer
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # Train and evaluate model
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        start_time = time.time()

        for i in range(TrainIteration):
            batch = mnist.train.next_batch(BatchSize)
            if i % DisplayFreq == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x: batch[0], y_: batch[1], keep_prob: 1.0})
                print('Step %d, time %.2fs, training accuracy %.4f' % (i, time.time() - start_time, train_accuracy))
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

        print('Test accuracy %.4f' % accuracy.eval(feed_dict={
            x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


if __name__ == '__main__':
    main()
