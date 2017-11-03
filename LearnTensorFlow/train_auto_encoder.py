#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
import matplotlib.pyplot as plt

from libs.models.auto_encoder import AdditiveGaussianNoiseAutoEncoder
from libs.utils.mnist import load_data

__author__ = 'fyabc'


def show_image(image_list, image_recons_list):
    N = len(image_list)
    for i, (image, image_recons) in enumerate(zip(image_list, image_recons_list)):
        image = image.reshape((28, 28))
        image_recons = image_recons.reshape((28, 28))

        plt.subplot(2, N, i + 1)
        plt.imshow(image, cmap='gray')
        plt.subplot(2, N, N + i + 1)
        plt.imshow(image_recons, cmap='gray')

    plt.show()


def test_recons(auto_encoder, X_test, start_index=0, end_index=0):
    """Reconstruct and show some images."""
    image_list = X_test[start_index: end_index + 1]
    image_recons_list = auto_encoder.reconstruct(image_list)
    show_image(image_list, image_recons_list)


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

    test_recons(auto_encoder, X_test, 0, 5)


if __name__ == '__main__':
    main()
