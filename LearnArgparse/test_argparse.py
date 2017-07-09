#! /usr/bin/python
# -*- coding: utf-8 -*-

from argparse import ArgumentParser

__author__ = 'fyabc'


def main(args=None):
    parser = ArgumentParser(
        fromfile_prefix_chars='@',
    )

    parser.add_argument('-a', action='store', type=int, default=10)

    args = parser.parse_args(args)

    print(args)


if __name__ == '__main__':
    main(['-a', '7', '-a', '9', '@../LearnArgparse/args.txt', ])
