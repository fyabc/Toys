#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Automatic math puzzle solver.

Input example::

    # Equation type
    @type mul

    # Puzzle content
    @puzzle
          ? 5
        * ? ?
        -----
          7 ?
      ? 0 ?
      -------
      2 ? ? ?
    @endpuzzle

    # Extra constraints
"""

import argparse
from pathlib import Path
from typing import Sequence

from line_profiler_pycharm import profile


def _parse_element(ch):
    if '0' <= ch <= '9':
        return ch
    if ch == '?':
        return '?'
    raise ValueError(f'Unsupported character {ch}')


def _check_line(line, value: int):
    value_s = str(value)
    if len(line) != len(value_s):
        return False
    for expected, real in zip(line, value_s):
        if expected != '?' and expected != real:
            return False
    return True


class Puzzle:
    def __init__(self, raw_str: str, pz_type: str, eq_lines: Sequence[str]):
        self.raw_str = raw_str
        self.type = pz_type
        self.eq_lines = eq_lines

        self._parse_equation()

    @classmethod
    def from_string(cls, raw_str):
        pz_type = None
        lines = [line.strip() for line in raw_str.splitlines()]
        eq_lines = []
        inside_eq = False
        for line in lines:
            if not line:
                continue
            if line.startswith('@type'):
                pz_type = line.split()[1]
                continue
            if line.startswith('@content'):
                inside_eq = True
                continue
            if line.startswith('@endcontent'):
                inside_eq = False
                continue
            if inside_eq:
                eq_lines.append(line)

        if pz_type == 'mul':
            pz_cls = MulPuzzle
        else:
            raise ValueError(f'Unknown puzzle type {pz_type}')
        return pz_cls(raw_str, pz_type, eq_lines)

    def _parse_equation(self):
        raise NotImplementedError()

    def to_seq(self):
        """Convert the puzzle to a sequence.

        Example for multiply equation:
            ? 5
          * ? ?
          -----
            7 ?
        ? 0 ?
        -------
        2 ? ? ?

        => [*, [[?, 5], [?, ?], [7, ?], [?, 0, ?], [2, ?, ?, ?]]]
        """
        raise NotImplementedError()

    def solve(self, get_all: bool = True):
        raise NotImplementedError()


class MulPuzzle(Puzzle):
    def _parse_equation(self):
        self.multiplicand = [
            _parse_element(ch) for ch in self.eq_lines[0].split()
        ]
        self.multiplier = [
            _parse_element(ch) for ch in self.eq_lines[1].split()[1:]
        ]
        self.part_results = []
        for line in self.eq_lines[3:-2]:
            part_result = [_parse_element(ch) for ch in line.split()]
            self.part_results.append(part_result)
        self.product = [
            _parse_element(ch) for ch in self.eq_lines[-1].split()
        ]

    def to_seq(self):
        return [
            '*', [
                self.multiplicand[:],
                self.multiplier[:],
                *self.part_results,
                self.product[:],
            ]]

    def solve(self, get_all: bool = True):
        n_candidates = self.multiplicand.count('?') + self.multiplier.count('?')

        guess = []

        def _check():
            guess_multiplicand_s = ''
            guess_multiplier_s = ''
            guess_i = 0
            for e in self.multiplicand:
                if e == '?':
                    e2 = str(guess[guess_i])
                    guess_i += 1
                else:
                    e2 = e
                guess_multiplicand_s += e2
            for e in self.multiplier:
                if e == '?':
                    e2 = str(guess[guess_i])
                    guess_i += 1
                else:
                    e2 = e
                guess_multiplier_s += e2
            guess_multiplicand = int(guess_multiplicand_s)
            guess_multiplier = int(guess_multiplier_s)

            if not _check_line(self.product, guess_multiplicand * guess_multiplier):
                return None, None
            for i, ch in enumerate(reversed(guess_multiplier_s)):
                if not _check_line(self.part_results[i], guess_multiplicand * int(ch)):
                    return None, None
            return guess_multiplicand, guess_multiplier

        found_result = False

        def _do_solve():
            nonlocal found_result

            if found_result and not get_all:
                return

            if len(guess) == n_candidates:
                result_multiplicand, result_multiplier = _check()
                if result_multiplicand is not None:
                    print(f'{result_multiplicand} * {result_multiplier} = {result_multiplicand * result_multiplier}')
                    found_result = True
                return
            for i in range(10):
                guess.append(i)
                _do_solve()
                guess.pop()

        _do_solve()


@profile
def parse_puzzle(file_handle) -> Puzzle:
    value = file_handle.read()
    puzzle = Puzzle.from_string(value)
    return puzzle


def main():
    parser = argparse.ArgumentParser(description='Math puzzle solver.')
    parser.add_argument('input_file', type=Path, help='Input puzzle filename')

    args = parser.parse_args()

    with args.input_file.open('r', encoding='utf-8') as f:
        puzzle = parse_puzzle(f)

    puzzle.solve()


if __name__ == '__main__':
    main()
