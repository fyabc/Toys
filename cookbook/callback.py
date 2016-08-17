#! /usr/bin/python3
# -*- encoding: utf-8 -*-

from queue import Queue
from functools import wraps

__author__ = 'fyabc'


def applyAsync(func, args=None, kwargs=None, *, callback=None):
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    result = func(*args, **kwargs)

    if callback is not None:
        callback(result)
    return result


class Async:
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs


def inlineAsync(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        f = func(*args, **kwargs)

        resultQueue = Queue()
        resultQueue.put(None)
        while True:
            result = resultQueue.get()
            try:
                a = f.send(result)
                applyAsync(a.func, a.args, a.kwargs, callback=resultQueue.put)
            except StopIteration:
                break
    return wrapper


def test():
    def add(x, y):
        return x + y

    @inlineAsync
    def testInlineAsync():
        r = yield Async(add, 2, 3)
        print(r)

        r = yield Async(add, 'Hello, ', 'world!')
        print(r)

        for i in range(10):
            r = yield Async(add, i, i)
            print(r)

        print('Goodbye')

    testInlineAsync()


if __name__ == '__main__':
    test()
