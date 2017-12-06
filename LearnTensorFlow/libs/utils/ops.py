#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.contrib as tfc

__author__ = 'fyabc'


def batch_norm(epsilon=1e-5, momentum=0.9, name='batch_norm'):
    def batch_norm_inner(x, train=True):
        return tfc.layers.batch_norm(
            x,
            decay=momentum,
            updates_collections=None,
            epsilon=epsilon,
            scale=True,
            is_training=train,
            scope=name,
        )

    return batch_norm_inner


def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""

    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat([x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], axis=3)


def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name='conv2d'):
    with tf.variable_scope(name):
        kernel = tf.get_variable('weights', [k_h, k_w, input_.get_shape()[-1], output_dim],
                                 initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, kernel, strides=[1, d_h, d_w, 1], padding='SAME')
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)

        return conv


def deconv2d(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name='deconv_2d', ret_weight=False):
    with tf.variable_scope(name):
        kernel = tf.get_variable('weights', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                                 initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(input_, kernel, output_shape=output_shape, strides=[1, d_h, d_w, 1])
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.nn.bias_add(deconv, biases)

        if ret_weight:
            return deconv, kernel, biases
        else:
            return deconv


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, ret_weight=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "linear"):
        matrix = tf.get_variable("weights", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("biases", [output_size],
                               initializer=tf.constant_initializer(bias_start))
        if ret_weight:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias


__all__ = [
    'batch_norm',
    'conv_cond_concat',
    'conv2d',
    'deconv2d',
    'linear',
]
