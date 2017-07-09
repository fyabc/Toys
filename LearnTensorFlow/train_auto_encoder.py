#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf

from libs.models.auto_encoder import AdditiveGaussianNoiseAutoEncoder
from libs.utils.mnist import load_data

__author__ = 'fyabc'


def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)

    return X_train, X_test


def random_batch(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index: start_index + batch_size]


def main():
    mnist = load_data()
    X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)

    n_samples = int(mnist.train.num_examples)
    epochs = 30
    batch_size = 128
    display_freq = 1

    auto_encoder = AdditiveGaussianNoiseAutoEncoder(
        n_in=784,
        n_hidden_list=[200],
        activation=tf.nn.softplus,
        optimizer=tf.train.AdamOptimizer(0.001),
        scale=0.01,
    )

    for epoch in range(epochs):
        avg_cost = 0.
        total_batch = n_samples // batch_size

        for i in range(total_batch):
            batch_xs = random_batch(X_train, batch_size)

            cost = auto_encoder.partial_fit(batch_xs)
            avg_cost += cost / n_samples * batch_size

        if epoch % display_freq == 0:
            print('Epoch: {:04d} cost={:.9f}'.format(epoch, avg_cost))

    print('Total cost: {:.9f}'.format(auto_encoder.total_cost(X_test)))

if __name__ == '__main__':
    main()
