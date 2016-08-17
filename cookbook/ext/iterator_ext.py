#! /usr/bin/python3
# -*- encoding: utf-8 -*-

from collections import Iterable

__author__ = 'fyabc'


def flatten(items, ignoreTypes=(str, bytes)):
    for item in items:
        if isinstance(item, Iterable) and not isinstance(item, ignoreTypes):
            yield from flatten(item, ignoreTypes)
        else:
            yield item


def test():
    items = [1, 2, [3, 4, [5, 6], 7], 8, range(9, 11), ('11', '12')]
    for item in flatten(items):
        print(item)


if __name__ == '__main__':
    test()
