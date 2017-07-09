#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from .constants import floatX

__author__ = 'fyabc'


def xavier_init(n_in, n_out, constant=1):
    """Xavier initializer."""

    low = -constant * np.sqrt(6.0 / (n_in + n_out))
    high = -low
    return tf.random_uniform([n_in, n_out], minval=low, maxval=high, dtype=floatX)


__all__ = [
    'xavier_init',
]
