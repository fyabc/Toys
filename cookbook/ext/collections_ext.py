#! /usr/bin/python
# -*- encoding: utf-8 -*-

import heapq

__author__ = 'fyabc'


class PriorityQueue:
    """

    """

    def __init__(self):
        self._queue = []
        self._index = 0

    def __bool__(self):
        return bool(self._queue)

    def __len__(self):
        return len(self._queue)

    def push(self, item, priority=0):
        heapq.heappush(self._queue, (-priority, self._index, item))
        self._index += 1

    def pop(self):
        return heapq.heappop(self._queue)[-1]


def dedupe(iterable, key=None):
    """
    dedupe the iterable.

    :param iterable: an iterable to dedupe.
    :param key: a callable object to generate the comparison key (default is None).
    :return a generator to yield deduped items of iterable.
    """

    seen = set()
    for item in iterable:
        val = item if key is None else key(item)
        if val not in seen:
            yield item
            seen.add(val)


# Tests.


def __testPQueue():
    q = PriorityQueue()
    q.push('foo', 1)
    q.push('bar', 5)
    q.push('spam', 4)
    q.push('grok', 1)

    while q:
        print(q.pop())


def __test():
    __testPQueue()


if __name__ == '__main__':
    __test()
