#! /usr/bin/python3
# -*- encoding: utf-8 -*-

__author__ = 'fyabc'


def permute(n, k):
    if k > n > 0:
        return 0

    result = 1
    for i in range(n, n - k, -1):
        result *= i
    return result


def fac(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result


def test():
    print(permute(10, 3), fac(10))


if __name__ == '__main__':
    test()
