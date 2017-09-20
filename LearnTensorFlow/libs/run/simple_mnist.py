#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import summary as ts

from ..utils.mnist import *

__author__ = 'fyabc'


def main():
    # Get MNIST dataset.
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

    # Build softmax regressions.

    with tf.name_scope('input'):
        x, y_ = input_placeholders()

    # Model parameters.
    with tf.name_scope('params'):
        W = tf.Variable(tf.zeros([ImageSize, LabelSize], floatX), name='W')
        b = tf.Variable(tf.zeros([LabelSize], floatX), name='b')

    # Get model output.
    # [NOTE] `matmul(x, W)` is a small trick of swap `x` and `W`
    # to deal with `x` being a 2D tensor with multiple inputs.
    # [NOTE] In Python 3.5 or later, can use `x @ W`.
    y = tf.matmul(x, W) + b

    # Training.

    # This is numerically unstable, apply softmax.
    # cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)), axis=[1]))

    # todo: check it with source code
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    # Training function.
    train_step = tf.train.GradientDescentOptimizer(LearningRate).minimize(cross_entropy)

    # Session
    sess = tf.InteractiveSession()

    g = sess.graph

    # Summary
    ts.histogram('W', W)
    merged_summary_op = ts.merge_all()
    summary_writer = tf.summary.FileWriter('./output/simple_mnist_logs', sess.graph)

    # Initialize and train
    tf.global_variables_initializer().run()

    b_summary_save = False

    for step in range(TrainIteration):
        batch_xs, batch_ys = mnist.train.next_batch(BatchSize)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

        if b_summary_save:
            summary_str = sess.run(merged_summary_op)
            summary_writer.add_summary(summary_str, step)

    for op in g.get_operations():
        print('Op:', op.name, end=' ')
        for t in op.inputs:
            print(t.shape, end=' ')
        print('| ', end='')
        for t in op.outputs:
            print(t.shape, end=' ')
        print()

    # Evaluation
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.arg_max(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, floatX))

    print('Accuracy:', sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
