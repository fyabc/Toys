#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Solve math puzzle problem 18 and 19.

Problem 18::

    已知一个正整数乘法竖式中包含0~9的10个数字，求积的最小值。

Solution: 18 * 39 = 702

Problem 19::

    已知一个正整数除法竖式恰好整除，且竖式中包含0~9的10个数字，求被除数的最小值（计入最底部的0）。

Solution: 456 / 19 = 24
"""


def gen_divisors(n):
    i = 1
    while i <= n:
        if n % i == 0:
            yield i, n // i
        i += 1


def count_numbers_in_mul_equation(a: int, b: int) -> set:
    numbers = set()
    numbers.update(str(a))

    str_b = str(b)
    numbers.update(str_b)

    for digit_b in str_b:
        if digit_b == '0':
            continue
        digit_mul = a * int(digit_b)
        numbers.update(str(digit_mul))
    numbers.update(str(a * b))

    return numbers


def _strip0(n: str):
    """
    '123' -> '123'
    '034040' -> '34040'
    '0' -> '0'
    """

    n_strip = n.lstrip('0')
    if not n_strip:
        n_strip = '0'
    return n_strip


def count_numbers_in_div_equation(a: int, d: int) -> set:
    """

    >>> sorted(count_numbers_in_div_equation(702, 18))
    ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    >>> sorted(count_numbers_in_div_equation(702, 39))
    ['0', '1', '2', '3', '7', '8', '9']
    """
    numbers = set()
    numbers.update(str(a))
    numbers.update(str(d))

    q = a // d
    str_q = str(q)
    numbers.update(str_q)

    str_a = str(a)
    str_current_a = ''
    current_pos = 0
    for new_pos, digit_q in enumerate(str_q, start=len(str_a) - len(str_q) + 1):
        str_current_a += str_a[current_pos:new_pos]
        current_pos = new_pos

        if digit_q == '0':
            continue

        numbers.update(str_current_a)

        digit_mul = d * int(digit_q)
        numbers.update(str(digit_mul))

        current_a = int(_strip0(str_current_a))
        current_a %= d
        str_current_a = str(current_a)

    numbers.add('0')

    return numbers


def main():
    n = 1
    found = False
    while not found:
        for a, b in gen_divisors(n):
            numbers = count_numbers_in_mul_equation(a, b)
            if len(numbers) == 10:
                print('Found: {} * {} = {}'.format(a, b, n))
                found = True
                break
        n += 1

    n = 1
    found = False
    while not found:
        for a, b in gen_divisors(n):
            numbers = count_numbers_in_div_equation(n, a)
            if len(numbers) == 10:
                print('Found: {} / {} = {}'.format(n, a, b))
                found = True
                break
        n += 1


if __name__ == '__main__':
    main()
