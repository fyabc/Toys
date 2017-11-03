#! /usr/bin/python
# -*- coding: utf-8 -*-

import operator
from functools import reduce
from itertools import islice
import random

import matplotlib.pyplot as plt

__author__ = 'fyabc'

hypos = {
    1, 2, 4, 7, 8, 16, 32, 56, 64, 73, 128, 146, 256, 292, 448, 511, 512, 513, 1024, 1026, 2048, 2052, 3584, 3591, 4096,
    4104, 8192, 8208, 16384, 16416, 28672, 28728, 32768, 32832, 37376, 37449, 65536, 65664, 74752, 74898, 131072,
    131328, 149504, 149796, 229376, 229824, 261632, 262143,
}


def get_set(k):
    return reduce(operator.or_, random.sample(hypos, k))


def iter_count_hypos(k, n=None):
    s = set()
    if n is None:
        while True:
            s.add(get_set(k))
            yield len(s)
    else:
        for _ in range(n):
            s.add(get_set(k))
            yield len(s)


def main():
    k = 6
    interval = 500
    n = 500000

    it = iter_count_hypos(k)

    xs = list(range(0, n + 1, interval))
    ys = list(islice(it, 0, n + 1, interval))

    print(ys[-1])

    plt.plot(xs, ys)
    plt.show()


if __name__ == '__main__':
    main()
