#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np


def scale_pos_height(pos, self):
    """Scale vertices coordinates, scale height to [0, 1].
    Width value will /= self.height.
    """
    return pos / self.height


def scale_point(p, width, height):
    if p.ndim == 1:
        p[0] *= (width / height)
    else:
        p[:, 0] *= (width / height)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def all_distances(p, vertices):
    delta = vertices - p
    return np.hypot(delta[:, 0], delta[:, 1])


def distance_ls(p1, p2, p):
    """Distance to a line segment."""
    p_rel = p - p1
    p2_rel = p2 - p1

    length = np.sum(np.square(p2_rel))
    if length < 1e-6:
        return np.hypot(p_rel[:, 0], p_rel[:, 1])
    dot = np.dot(p_rel, p2_rel)
    proj = dot / length

    target = p1 + proj[:, None] * p2_rel
    target[proj < 0] = p1
    target[proj > 1] = p2
    delta = p - target
    return np.hypot(delta[:, 0], delta[:, 1])


def exp_2_dist(distances, decay_rate, scale):
    return np.exp(-distances ** 2 * decay_rate) ** 2 * scale


def normalize(elevations, lo=0., hi=1.):
    """Normalize to 0~1, then square root to smooth."""
    np.add(lo, (elevations - elevations.min()) * ((hi - lo) / np.ptp(elevations)), out=elevations)


def nearest_vertex(p, vertices):
    return np.argmin(all_distances(p, vertices))
