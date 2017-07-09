#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from .model import Model
from ..utils.constants import floatX

__author__ = 'fyabc'


class MLP(Model):
    def __init__(self, n_in, n_hidden_list, n_out,
                 dropout=0.75,
                 optimizer=None):
        self.n_in = n_in
        self.n_hidden_list = n_hidden_list
        self.n_out = n_out
        self.dropout = dropout
        self.optimizer = tf.train.AdagradOptimizer(0.3) if optimizer is None else optimizer

        self.parameters = {}
        self._init_weights()

        # todo

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _init_weights(self):
        for layer_id, n_hidden in enumerate(self.n_hidden_list):
            if layer_id == 0:
                shape = self.n_in, n_hidden
            else:
                shape = self.n_hidden_list[layer_id - 1], n_hidden
            self.parameters['w{}'.format(layer_id)] = tf.Variable(tf.truncated_normal(shape, stddev=0.1, dtype=floatX))
            self.parameters['b{}'.format(layer_id)] = tf.Variable(tf.zeros([n_hidden], dtype=floatX))

        self.parameters['w_softmax'] = tf.Variable(tf.zeros([self.n_hidden_list[-1], self.n_out], dtype=floatX))
        self.parameters['b_softmax'] = tf.Variable(tf.zeros([self.n_out], dtype=floatX))
