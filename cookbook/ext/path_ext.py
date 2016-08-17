#! /usr/bin/python3
# -*- encoding: utf-8 -*-

import os

__author__ = 'fyabc'


def relPath(root, *paths):
    """
    transform a relative path to the absolute path.
    :param root: the root of real path, should be __file__ of caller.
    :param paths: a list of paths to the destination.
    :return: the absolute path of the destination.
    """
    return os.path.join(os.path.dirname(root), *paths)
