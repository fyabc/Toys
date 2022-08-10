#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Show equation."""

import argparse
import re


_PAT_EQU = re.compile(r'(\d+)([+\-*/])(\d+)')


def _len(n: int):
    """Digit length."""
    return len(str(n))


def _digits(n: int, reverse=False):
    if reverse:
        return reversed([int(c) for c in str(n)])
    return [int(c) for c in str(n)]


def _add_sub(a: int, op: str, b: int):
    c = a + b if op == '+' else a - b
    width = max([_len(a), _len(b), _len(c)]) + 1

    print(format(a, '>{}'.format(width)))
    print(op, end='')
    print(format(b, '>{}'.format(width - 1)))
    print('-' * width)
    print(format(c, '>{}'.format(width)))


def _mul(a: int, b: int):
    c = a * b
    width = max([_len(a), _len(b), _len(c)]) + 1

    print(format(a, '>{}'.format(width)))
    print('*', end='')
    print(format(b, '>{}'.format(width - 1)))
    print('-' * width)

    for i, d in enumerate(_digits(b, reverse=True)):
        part_sum = a * d
        print(format(part_sum, '>{}'.format(width - i)))

    print('-' * width)
    print(format(c, '>{}'.format(width)))


def _div(a: int, b: int):
    c = a // b

    width = _len(b) + 1 + _len(a)

    print(format(c, '>{}'.format(width)))
    print(format('-' * _len(a), '>{}'.format(width)))
    print('{}){}'.format(b, a))

    start_pos = _len(a) - _len(c) + 1
    part_a = int(str(a)[:start_pos])

    c_digits = _digits(c)
    for i in range(len(c_digits)):
        pos = start_pos + i
        d = c_digits[i]

        part_sum = b * d

        part_a -= part_sum
        if pos < _len(a):
            part_a = part_a * 10 + int(str(a)[pos])

        align_width = _len(b) + 1 + pos
        part_a_width = align_width if pos == _len(a) else align_width + 1

        if d != 0:
            print(format(part_sum, '>{}'.format(align_width)))
            print(' ' * (_len(b) + 1), end='')
            print('-' * _len(a))

        if i == len(c_digits) - 1 or c_digits[i + 1] != 0:
            print(format(part_a, '>{}'.format(part_a_width)))


def main():
    parser = argparse.ArgumentParser('Show equation.')
    parser.add_argument('equation', nargs='+', help='Equation string')

    args = parser.parse_args()
    args.equation = ''.join(args.equation)

    match = _PAT_EQU.fullmatch(args.equation)
    if match is None:
        print('ERROR: Invalid input equation, must be {!r}'.format(_PAT_EQU.pattern))
        exit(1)

    a, op, b = int(match.group(1)), match.group(2), int(match.group(3))

    if a < 0 or b < 0:
        print('ERROR: does not support negative values')
        exit(1)

    if op in {'+', '-'}:
        _add_sub(a, op, b)
    elif op == '*':
        _mul(a, b)
    elif op == '/':
        _div(a, b)


if __name__ == '__main__':
    main()
