#! /usr/bin/python3
# -*- encoding: utf-8 -*-

__author__ = 'fyabc'


def truncateAfter(string, substring, stripThis=True):
    index = string.rfind(substring)
    if index == -1:
        index = None
    else:
        if not stripThis:
            index += len(substring)
    return string[:index]


class RichStr(str):
    def truncateAfter(self, substring, stripThis=True):
        return truncateAfter(self, substring, stripThis)


def test():
    print(truncateAfter('text__txt', '__'))


if __name__ == '__main__':
    test()
