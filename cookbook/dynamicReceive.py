#! /usr/bin/python3
# -*- encoding: utf-8 -*-

__author__ = 'fyabc'


class Receiver:
    def __init__(self):
        pass

    def __getattr__(self, item):
        print('Get an item:', item)
        return item

    def __setattr__(self, key, value):
        print("Set an item:", key, "to value:", value)


def test():
    r = Receiver()
    a = r.obj
    r.obj = 10
    print('a =', repr(a))


if __name__ == '__main__':
    test()
