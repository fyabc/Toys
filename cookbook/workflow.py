#! /usr/bin/python3
# -*- encoding: utf-8 -*-

from collections import Iterator


__author__ = 'fyabc'


def getWorkflow(cells):
    """
    Create workflow in a function, cells a generators.
    :param cells: an iterable of work cells. Each cell must be a generator.
    :return: a workflow, input parameters, then output a generator.
    """

    if not isinstance(cells, Iterator):
        cells = iter(cells)

    def runnable(*args, **kwargs):
        result = next(cells)(*args, **kwargs)
        for cell in cells:
            result = cell(result)
        return result
    return runnable


def testWorkflow():
    import os
    import fnmatch
    import re
    from functools import partial

    def gFindFile(pattern, top):
        for path, dirList, fileList in os.walk(top):
            for name in fnmatch.filter(fileList, pattern):
                yield os.path.join(path, name)

    def gOpenFile(names):
        for name in names:
            with open(name, 'rt') as f:
                yield f

    def gConcatenate(iterators):
        for it in iterators:
            yield from it

    def gGrep(pattern, lines):
        pat = re.compile(pattern)
        for line in lines:
            if pat.search(line):
                yield line

    workflow = getWorkflow([gFindFile, gOpenFile, gConcatenate, partial(gGrep, r'^\s*def ')])

    defLines = workflow('*.py', 'C:/Users/v-yanfa/PycharmProjects')

    for line in defLines:
        print(line)


def test():
    testWorkflow()


if __name__ == '__main__':
    test()
