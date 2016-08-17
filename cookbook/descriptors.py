#! /usr/bin/python3
# -*- encoding: utf-8 -*-

__author__ = 'fyabc'


class Attribute:
    """

    """

    def __init__(self, name):
        self.name = name

    def __get__(self, instance, owner):
        if instance is None:
            return self
        else:
            return instance.__dict__[self.name]

    def __set__(self, instance, value):
        instance.__dict__[self.name] = value

    def __delete__(self, instance):
        del instance.__dict__[self.name]


class Integer(Attribute):
    def __set__(self, instance, value):
        if not isinstance(value, int):
            raise TypeError('Expected an integer, got %s' % value)
        super(Integer, self).__set__(instance, value)


def test():
    class IPoint:
        x = Integer('x')
        y = Integer('y')

        def __init__(self, x, y):
            self.x = x
            self.y = y

    try:
        p = IPoint(2, 3)
        p.x = 5
        del p.x
        p.x = 7
        p.y = 2.3
    except TypeError as e:
        print(e)


if __name__ == '__main__':
    test()
