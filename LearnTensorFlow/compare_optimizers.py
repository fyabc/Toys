#! /usr/bin/python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import tensorflow as tf

__author__ = 'fyabc'


def cliff(x, y):
    z = 2.5 * tf.nn.sigmoid(3 * (x + y)) + (x ** 2 + y ** 2) * 0.015

    return z


def square(x, y):
    return x ** 2 + y ** 2


def update(z, optimizer=tf.train.GradientDescentOptimizer):
    return optimizer(learning_rate=1.0).minimize(z)


def plot_value(xs, ys, zs):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xs, ys = np.meshgrid(xs, ys)

    ax.plot_surface(xs, ys, zs)


def plot_line(xs, ys, zs):
    plt.plot(xs, ys, zs=zs)


def main():
    x = tf.Variable(0.0, dtype=tf.float32, name='x')
    y = tf.Variable(0.0, dtype=tf.float32, name='y')

    functions = {
        'cliff': cliff,
        'square': square,
    }
    function_name = 'square'

    out = functions[function_name](x, y)
    update_op = update(out)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        h, w = 30, 30
        out_vals = np.zeros([w, h], dtype='float32')
        xs = np.linspace(-5.0, 5.0, w, dtype='float32')
        ys = np.linspace(-5.0, 5.0, w, dtype='float32')

        for i, x_val in enumerate(xs):
            for j, y_val in enumerate(ys):
                out_vals[i, j] = sess.run(out, feed_dict={x: x_val, y: y_val})

        plot_value(xs, ys, out_vals)

        sess.run([x.assign(4.0), y.assign(4.1)])

        xs, ys, zs = [], [], []
        for i in range(200):
            x_val, y_val, z, _ = sess.run([x, y, out, update_op])
            xs.append(x_val)
            ys.append(y_val)
            zs.append(z)

        plot_line(xs, ys, zs)

    plt.show()


if __name__ == '__main__':
    main()
