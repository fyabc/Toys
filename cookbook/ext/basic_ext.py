#! /usr/bin/python3
# -*- encoding: utf-8 -*-

from operator import lt, gt


__author__ = 'fyabc'


def frange(start, stop, step=1.0):
    x = start
    cmp = lt if step > 0 else gt
    while cmp(x, stop):
        yield x
        x += step


def unzip(zipped):
    head = next(zipped)

    result = tuple([elem] for elem in head)
    for tup in zipped:
        for i, elem in enumerate(tup):
            result[i].append(elem)

    return result


def test():
    z = zip([1, 3, 5, 7], [2, 4, 6, 8])
    print(unzip(z))


if __name__ == '__main__':
    test()
