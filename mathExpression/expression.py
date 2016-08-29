#! /usr/bin/python
# -*- encoding: utf-8 -*-

__author__ = 'fyabc'


class Expression:
    precedence = None

    def __init__(self):
        self.operands = []


class Variable(Expression):
    precedence = None

    def __init__(self, name=''):
        super(Variable, self).__init__()
        self.name = name


def test():
    pass


if __name__ == '__main__':
    test()
