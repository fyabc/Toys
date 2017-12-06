#! /usr/bin/python
# -*- coding: utf-8 -*-

import math

import tensorflow as tf

from ...utils.ops import *

__author__ = 'fyabc'


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


class Config:
    """
    sess: TensorFlow session
    batch_size: The size of batch. Should be specified before training.
    y_dim: (optional) Dimension of dim for y. [None]
    z_dim: (optional) Dimension of dim for Z. [100]
    gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
    df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
    gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
    dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
    c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        NOTE: this must be set according to the dataset. E.g., `data_X[0].shape[-1]`.
    """

    input_height = 108
    input_width = 108
    crop = True
    batch_size = 64
    sample_num = 64
    output_height = 64
    output_width = 64
    y_dim = None
    z_dim = 100
    gf_dim = 64
    df_dim = 64
    gfc_dim = 1024
    dfc_dim = 1024
    c_dim = 3
    dataset_name = 'default'
    input_fname_pattern = '*.jpg'
    checkpoint_dir = None
    sample_dir = None


class DCGAN:
    def __init__(self, is_training, config: Config):
        self.C = config
        self.is_training = is_training

        self.build_model()

    def build_model(self):
        # Batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        if not self.C.y_dim:
            self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        if not self.C.y_dim:
            self.g_bn3 = batch_norm(name='g_bn3')

        # One-hot label y, ([B], [y])
        if self.C.y_dim:
            self.y = tf.placeholder(tf.float32, [self.C.batch_size, self.C.y_dim], name='y')
        else:
            self.y = None

        # Real image X, ([B], [Hi/Ho], [Wi/Wo], [C])
        if self.C.crop:
            image_dims = [self.C.batch_size, self.C.output_height, self.C.output_width, self.C.c_dim]
        else:
            image_dims = [self.C.batch_size, self.C.input_height, self.C.input_width, self.C.c_dim]
        self.X = tf.placeholder(tf.float32, image_dims, name='real_images')

        # Random noise input z, ([B], [z])
        self.z = tf.placeholder(tf.float32, [None, self.C.z_dim], name='z')

        # Build discriminator and generator (run discriminator on real images and generated images).
        self.G = self.generator(self.z, self.y)
        self.D, self.D_logits = self.discriminator(self.X, self.y, reuse=False)
        self.sample = self.sampler(self.z, self.y)
        self.Dg, self.Dg_logits = self.discriminator(self.G, self.y, reuse=True)

        # todo

    def discriminator(self, image, y=None, reuse=False):
        """Discriminator.

        :param image: Real image, ([B], [Hi/Ho], [Wi/Wo], [C])
        :param y: One-hot labels, ([B], [y])
        :param reuse: Reuse scope or not
        :return: Two tensors
            last hidden state (after sigmoid), last hidden state (before sigmoid)
        """

        with tf.variable_scope("discriminator", reuse=reuse):
            if not self.C.y_dim:
                h0 = tf.nn.leaky_relu(conv2d(image, self.C.df_dim, name='d_h0_conv'))
                h1 = tf.nn.leaky_relu(self.d_bn1(conv2d(h0, self.C.df_dim * 2, name='d_h1_conv')))
                h2 = tf.nn.leaky_relu(self.d_bn2(conv2d(h1, self.C.df_dim * 4, name='d_h2_conv')))
                h3 = tf.nn.leaky_relu(self.d_bn3(conv2d(h2, self.C.df_dim * 8, name='d_h3_conv')))
                h4 = linear(tf.reshape(h3, [self.C.batch_size, -1]), 1, 'd_h4_linear')

                return tf.nn.sigmoid(h4), h4
            else:
                yb = tf.reshape(y, [self.C.batch_size, 1, 1, self.C.y_dim])
                x = conv_cond_concat(image, yb)

                h0 = tf.nn.leaky_relu(conv2d(x, self.C.c_dim + self.C.y_dim, name='d_h0_conv'))
                h0 = conv_cond_concat(h0, yb)

                h1 = tf.nn.leaky_relu(self.d_bn1(conv2d(h0, self.C.df_dim + self.C.y_dim, name='d_h1_conv')))
                h1 = tf.reshape(h1, [self.C.batch_size, -1])
                h1 = tf.concat([h1, y], 1)

                h2 = tf.nn.leaky_relu(self.d_bn2(linear(h1, self.C.dfc_dim, 'd_h2_linear')))
                h2 = tf.concat([h2, y], 1)

                h3 = linear(h2, 1, 'd_h3_linear')

                return tf.nn.sigmoid(h3), h3

    def generator(self, z, y=None):
        """Generator

        :param z: Random noise input, ([B], [z])
        :param y: One-hot labels, ([B], [y])
        :return: Tensor:
            generated images, ([B], [Hi/Ho], [Wi/Wo], [C])
        """

        with tf.variable_scope("generator"):
            if not self.C.y_dim:
                s_h, s_w = self.C.output_height, self.C.output_width
                s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
                s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
                s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
                s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

                # project `z` and reshape
                self.z_, self.h0_w, self.h0_b = linear(
                    z, self.C.gf_dim * 8 * s_h16 * s_w16, 'g_h0_linear', ret_weight=True)

                self.h0 = tf.reshape(self.z_, [-1, s_h16, s_w16, self.C.gf_dim * 8])
                h0 = tf.nn.relu(self.g_bn0(self.h0))

                self.h1, self.h1_w, self.h1_b = deconv2d(
                    h0, [self.C.batch_size, s_h8, s_w8, self.C.gf_dim * 4], name='g_h1', ret_weight=True)
                h1 = tf.nn.relu(self.g_bn1(self.h1))

                h2, self.h2_w, self.h2_b = deconv2d(
                    h1, [self.C.batch_size, s_h4, s_w4, self.C.gf_dim * 2], name='g_h2', ret_weight=True)
                h2 = tf.nn.relu(self.g_bn2(h2))

                h3, self.h3_w, self.h3_b = deconv2d(
                    h2, [self.C.batch_size, s_h2, s_w2, self.C.gf_dim * 1], name='g_h3', ret_weight=True)
                h3 = tf.nn.relu(self.g_bn3(h3))

                h4, self.h4_w, self.h4_b = deconv2d(
                    h3, [self.C.batch_size, s_h, s_w, self.C.c_dim], name='g_h4', ret_weight=True)

                return tf.nn.tanh(h4)
            else:
                s_h, s_w = self.C.output_height, self.C.output_width
                s_h2, s_h4 = int(s_h / 2), int(s_h / 4)
                s_w2, s_w4 = int(s_w / 2), int(s_w / 4)

                # yb = tf.expand_dims(tf.expand_dims(y, 1),2)
                yb = tf.reshape(y, [self.C.batch_size, 1, 1, self.C.y_dim])
                z = tf.concat([z, y], 1)

                h0 = tf.nn.relu(self.g_bn0(linear(z, self.C.gfc_dim, 'g_h0_linear')))
                h0 = tf.concat([h0, y], 1)

                h1 = tf.nn.relu(self.g_bn1(linear(h0, self.C.gf_dim * 2 * s_h4 * s_w4, 'g_h1_linear')))
                h1 = tf.reshape(h1, [self.C.batch_size, s_h4, s_w4, self.C.gf_dim * 2])
                h1 = conv_cond_concat(h1, yb)

                h2 = tf.nn.relu(self.g_bn2(
                    deconv2d(h1, [self.C.batch_size, s_h2, s_w2, self.C.gf_dim * 2], name='g_h2')))
                h2 = conv_cond_concat(h2, yb)

                return tf.nn.sigmoid(deconv2d(h2, [self.C.batch_size, s_h, s_w, self.C.c_dim], name='g_h3'))

    def sampler(self, z, y=None):
        """Run the generator in inference mode.

        :param z: Random noise input, ([B], [z])
        :param y: One-hot labels, ([B], [y])
        :return: Tensor:
            generated images, ([B], [Hi/Ho], [Wi/Wo], [C])
        """

        with tf.variable_scope("generator", reuse=True):
            if not self.C.y_dim:
                s_h, s_w = self.C.output_height, self.C.output_width
                s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
                s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
                s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
                s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

                # project `z` and reshape
                h0 = tf.reshape(
                    linear(z, self.C.gf_dim * 8 * s_h16 * s_w16, 'g_h0_linear'),
                    [-1, s_h16, s_w16, self.C.gf_dim * 8])
                h0 = tf.nn.relu(self.g_bn0(h0, train=False))

                h1 = deconv2d(h0, [self.C.batch_size, s_h8, s_w8, self.C.gf_dim * 4], name='g_h1')
                h1 = tf.nn.relu(self.g_bn1(h1, train=False))

                h2 = deconv2d(h1, [self.C.batch_size, s_h4, s_w4, self.C.gf_dim * 2], name='g_h2')
                h2 = tf.nn.relu(self.g_bn2(h2, train=False))

                h3 = deconv2d(h2, [self.C.batch_size, s_h2, s_w2, self.C.gf_dim * 1], name='g_h3')
                h3 = tf.nn.relu(self.g_bn3(h3, train=False))

                h4 = deconv2d(h3, [self.C.batch_size, s_h, s_w, self.C.c_dim], name='g_h4')

                return tf.nn.tanh(h4)
            else:
                s_h, s_w = self.C.output_height, self.C.output_width
                s_h2, s_h4 = int(s_h / 2), int(s_h / 4)
                s_w2, s_w4 = int(s_w / 2), int(s_w / 4)

                # yb = tf.reshape(y, [-1, 1, 1, self.y_dim])
                yb = tf.reshape(y, [self.C.batch_size, 1, 1, self.C.y_dim])
                z = tf.concat([z, y], 1)

                h0 = tf.nn.relu(self.g_bn0(linear(z, self.C.gfc_dim, 'g_h0_linear'), train=False))
                h0 = tf.concat([h0, y], 1)

                h1 = tf.nn.relu(self.g_bn1(linear(h0, self.C.gf_dim * 2 * s_h4 * s_w4, 'g_h1_linear'), train=False))
                h1 = tf.reshape(h1, [self.C.batch_size, s_h4, s_w4, self.C.gf_dim * 2])
                h1 = conv_cond_concat(h1, yb)

                h2 = tf.nn.relu(self.g_bn2(
                    deconv2d(h1, [self.C.batch_size, s_h2, s_w2, self.C.gf_dim * 2], name='g_h2'), train=False))
                h2 = conv_cond_concat(h2, yb)

                return tf.nn.sigmoid(deconv2d(h2, [self.C.batch_size, s_h, s_w, self.C.c_dim], name='g_h3'))

