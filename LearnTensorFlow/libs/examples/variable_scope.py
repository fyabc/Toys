#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

__author__ = 'fyabc'


def test_variable_scope():
    with tf.variable_scope('lstm_0'):
        x = tf.get_variable('x', shape=[3, 4])
        print('x:', x.name)

    with tf.variable_scope('lstm_0', reuse=True):
        x = tf.get_variable('x')
        print('x:', x.name)

        y = tf.get_variable('y', shape=[4, 5])
        print('y:', y.name)
