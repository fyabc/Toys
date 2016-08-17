#! /usr/bin/python3
# -*- encoding: utf-8 -*-

from collections import defaultdict
from operator import add, sub, mul, truediv
import time

__author__ = 'fyabc'


def permute2(size):
    for i in range(size):
        for j in range(size):
            if i == j:
                continue
            yield i, j


def fac(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result


def multiPermuteNum(*args):
    result = fac(len(args))
    reps = defaultdict(int)
    for e in args:
        reps[e] += 1
    for v in reps.values():
        result //= fac(v)
    return result


class MyOperator:
    def __init__(self, op, name, exchangeable=False):
        self.op = op
        self.name = name
        self.exchangeable = exchangeable

    def __call__(self, lhs, rhs):
        return self.op(lhs, rhs)


class MyTester:
    DefaultOpList = (
        MyOperator(add, '+', True),
        MyOperator(sub, '-'),
        MyOperator(mul, '*', True),
        MyOperator(truediv, '/')
    )

    def __init__(self, opList=DefaultOpList, target=24, firstResult=True):
        self.opList = opList
        self.target = target
        self.firstResult = firstResult
        self.epsilon = 1e-6

        self.found = False
        self.currentNumList = tuple()
        self.record = []

        self.allRecords = {}

    def test(self, numList):
        self.found = False
        self.currentNumList = tuple(numList)
        self.record = []
        self._test(tuple([float(num) for num in numList]))
        return self.found

    def _test(self, numList):
        if self.found and self.firstResult:
            return

        if len(numList) == 1:
            if abs(numList[0] - self.target) < self.epsilon:
                if self.currentNumList not in self.allRecords:
                    self.allRecords[self.currentNumList] = self.record[:]
                self.found = True
            return

        for i, j in permute2(len(numList)):
            for op in self.opList:
                if i > j and op.exchangeable:
                    continue

                newNumList = [numList[index] for index in range(len(numList)) if index != i and index != j]
                try:
                    result = op(numList[i], numList[j])
                    self.record.append([numList[i], numList[j], op.name])
                except ZeroDivisionError:
                    continue
                newNumList.append(result)
                self._test(tuple(newNumList))
                self.record.pop()


def main():
    tester = MyTester()

    ranges = [
        (1, 13),
        (1, 13),
        (1, 13),
        (1, 13),
    ]

    totalCases = 1
    for r in ranges:
        totalCases *= r[1] - r[0] + 1

    timeBefore = time.time()

    uniqueCases = 0
    passedCases = 0
    for a in range(ranges[0][0], ranges[0][1] + 1):
        for b in range(max(a, ranges[1][0]), ranges[1][1] + 1):
            for c in range(max(b, ranges[2][0]), ranges[2][1] + 1):
                for d in range(max(c, ranges[3][0]), ranges[3][1] + 1):
                    found = tester.test([a, b, c, d])
                    uniqueCases += 1
                    if found:
                        passedCases += multiPermuteNum(a, b, c, d)

    timeAfter = time.time()

    print('Total cases:', totalCases)
    print('Unique cases:', uniqueCases)
    print('Time passed: %.3fs' % (timeAfter - timeBefore,))
    print('Time per case: %.3fs' % ((timeAfter - timeBefore) / uniqueCases))

    uniquePassedCases = len(tester.allRecords)
    print('Passed cases:', passedCases)
    print('Unique passed cases:', uniquePassedCases)
    print('Pass rate: %.3f' % (passedCases / totalCases))
    print('Unique pass rate: %.3f' % (uniquePassedCases / uniqueCases))

    for case in sorted(tester.allRecords):
        print(case, '\t', tester.allRecords[case], sep='')


if __name__ == '__main__':
    main()
