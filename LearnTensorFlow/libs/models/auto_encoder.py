#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from .model import Model
from ..utils.constants import floatX
from ..utils.initializer import xavier_init

__author__ = 'fyabc'


class AdditiveGaussianNoiseAutoEncoder(Model):
    def __init__(self, n_in, n_hidden_list,
                 activation=tf.nn.softplus,
                 optimizer=None,
                 scale=0.1):
        """

        :param n_in: Input dim.
        :param n_hidden_list: List of hidden dims of each layer.
        :param activation: Activation function.
        :param optimizer: TensorFlow optimizer instance, default is Adam optimizer.
        :param scale: Gaussian noise parameter.
        """

        optimizer = tf.train.AdamOptimizer() if optimizer is None else optimizer

        self.n_in = n_in
        self.n_hidden_list = n_hidden_list
        self.activation = activation
        self.training_scale = scale

        # Model parameters.
        self.parameters = {}
        self._init_weights()

        # Model placeholders.
        self.scale = tf.placeholder(floatX)
        self.x = tf.placeholder(floatX, [None, self.n_in])

        _hidden = self.x + scale * tf.random_normal([n_in])
        self.hidden_list = []

        for layer_id in range(len(n_hidden_list)):
            _hidden = self.activation(
                tf.matmul(_hidden, self.parameters['w{}'.format(layer_id)]) +
                self.parameters['b{}'.format(layer_id)]
            )
            self.hidden_list.append(_hidden)
        self.hidden = self.hidden_list[-1]

        self.reconstruction = tf.matmul(self.hidden_list[-1], self.parameters['w_rec']) + self.parameters['b_rec']

        # Cost and train operations.
        self.cost = 0.5 * tf.reduce_sum((self.reconstruction - self.x) ** 2.0)
        self.train_op = optimizer.minimize(self.cost)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _init_weights(self):
        for layer_id, n_hidden in enumerate(self.n_hidden_list):
            if layer_id == 0:
                shape = self.n_in, n_hidden
            else:
                shape = self.n_hidden_list[layer_id - 1], n_hidden
            self.parameters['w{}'.format(layer_id)] = tf.Variable(xavier_init(*shape))
            self.parameters['b{}'.format(layer_id)] = tf.Variable(tf.zeros([n_hidden], dtype=floatX))

        self.parameters['w_rec'] = tf.Variable(tf.zeros([self.n_hidden_list[-1], self.n_in], dtype=floatX))
        self.parameters['b_rec'] = tf.Variable(tf.zeros([self.n_in], dtype=floatX))

    def partial_fit(self, X):
        cost, _ = self.sess.run([self.cost, self.train_op],
                                feed_dict={self.x: X, self.scale: self.training_scale})
        return cost

    def total_cost(self, X):
        return self.sess.run(
            self.cost,
            feed_dict={self.x: X, self.scale: self.training_scale}
        )

    def transform(self, X):
        return self.sess.run(
            self.hidden,
            feed_dict={self.x: X, self.scale: self.training_scale}
        )

    def generate(self, hidden=None):
        if hidden is None:
            hidden = np.random.normal(size=self.parameters['b{}'.format(len(self.n_hidden_list) - 1)].shape)
        return self.sess.run(
            self.reconstruction,
            feed_dict={self.hidden: hidden}
        )

    def reconstruct(self, X):
        return self.sess.run(
            self.reconstruction,
            feed_dict={self.x: X, self.scale: self.training_scale}
        )
