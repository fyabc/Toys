#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from libs.models.cnn.alex_net import inference
from libs.utils.basic import time_tensorflow_run
from libs.utils.constants import floatX

__author__ = 'fyabc'


def run_benchmark():
    batch_size = 32
    num_batches = 100

    options = {
        'lrn': False,
    }

    with tf.Graph().as_default():
        image_size = 224
        images = tf.Variable(
            tf.random_normal(
                [batch_size, image_size, image_size, 3],
                dtype=floatX,
                stddev=1e-1,
            )
        )

        fc3, parameters = inference(images, options)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        # Forward benchmark
        time_tensorflow_run(sess, fc3, 'Forward', num_batches)

        # Backward benchmark
        loss = tf.nn.l2_loss(fc3)
        grad = tf.gradients(loss, parameters)
        time_tensorflow_run(sess, grad, 'Forward-backward', num_batches)


def main():
    run_benchmark()


if __name__ == '__main__':
    main()
