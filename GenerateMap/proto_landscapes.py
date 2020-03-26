#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Proto-landscapes."""

import operator
from collections import deque

import numpy as np

try:
    from . import math_utils as mu
except ImportError:
    import math_utils as mu


def _random_point(n, width, height) -> np.ndarray:
    return np.asarray([
        np.random.uniform(0, width / height, (n,)),
        np.random.uniform(0, 1, (n,)),
    ]).T


def noise_landscape(self, scale=1.):
    if scale > 0:
        r = 0, scale
    else:
        r = scale, 0
    return np.random.uniform(*r, self.elevations.shape)


def slope_landscape(self, scale=1., slope=50., a=None, b=None, c=None):
    elevations = np.zeros_like(self.elevations)
    if a is None:
        a = np.random.uniform(-1, 1)
    if b is None:
        b = np.random.uniform(-1, 1)
    if c is None:
        c = np.random.uniform(-0.5, 0.5)

    for v in range(self.n_vertices):
        x, y = self.vertices[v]
        _x, _y = x / self.width - 0.5, y / self.height - 0.5
        elevations[v] = scale * mu.sigmoid(slope * (a * _x + b * _y + c))
    return elevations


def _n_to_pos(n_or_pos, width, height) -> np.ndarray:
    if isinstance(n_or_pos, int):
        return _random_point(n_or_pos, width, height)
    if not isinstance(n_or_pos, np.ndarray):
        n_or_pos = np.asarray(n_or_pos)
    mu.scale_point(n_or_pos, width, height)
    return n_or_pos


def cone_landscape(self, scale=1., n_or_pos=5, decay_rate=5.):
    """Cone landscape, based on distance.

    :param self:
    :param scale:
    :param Union[int, tuple] n_or_pos: int or np.ndarray
        Number of cones or cone positions.
        (position must between 0 and 1)
    :param decay_rate:
    :return:
    """
    cones = _n_to_pos(n_or_pos, self.width, self.height)
    elevations = np.zeros_like(self.elevations)
    normalized_vertices = mu.scale_pos_height(self.vertices, self)
    for cone in cones:
        elevations += mu.exp_2_dist(mu.distance(cone, normalized_vertices), decay_rate, scale)
    return elevations


def line_landscape(self, scale=1., p1=None, p2=None, decay_rate=5.):
    """Line landscape, like "mountains".

    :param self:
    :param scale:
    :param Iterable p1:
    :param Iterable p2:
    :param decay_rate:
    :return:
    """
    if p1 is None:
        p1 = _random_point(1, self.width, self.height)[0]
    else:
        p1 = np.asarray(p1)
    mu.scale_point(p1, self.width, self.height)
    if p2 is None:
        p2 = _random_point(1, self.width, self.height)[0]
    else:
        p2 = np.asarray(p2)
    mu.scale_point(p2, self.width, self.height)

    elevations = np.zeros_like(self.elevations)
    normalized_vertices = mu.scale_pos_height(self.vertices, self)
    elevations += mu.exp_2_dist(mu.distance_ls(p1, p2, normalized_vertices), decay_rate, scale)
    return elevations


def plateau_landscape(self, scale=1., n_or_pos=5, edge_dist=0.2, decay_rate=5.0):
    """Plateau landscape, based on distance.

    :param self:
    :param scale:
    :param Union[int, tuple] n_or_pos: int or np.ndarray
        Number of plateaus or cone positions.
        (position must between 0 and 1)
    :param edge_dist: The distance of plateau edge (elevation drop to scale / 2)
    :param decay_rate
    :return:
    """
    plateaus = _n_to_pos(n_or_pos, self.width, self.height)
    elevations = np.zeros_like(self.elevations)
    normalized_vertices = mu.scale_pos_height(self.vertices, self)

    for plateau in plateaus:
        d = mu.distance(plateau, normalized_vertices)
        e = mu.sigmoid(decay_rate * (edge_dist - d)) * scale
        elevations += e
    return elevations


def plateau_noise_landscape(self, scale=1., n_or_pos=5, edge_dist=0.2, seed=1):
    """

    :param self:
    :param scale:
    :param Union[int, tuple] n_or_pos:
    :param edge_dist:
    :param int seed:
    :return:
    """
    rng = np.random.RandomState(seed)
    cones = _n_to_pos(n_or_pos, self.width, self.height)
    elevations = np.zeros_like(self.elevations)
    normalized_vertices = mu.scale_pos_height(self.vertices, self)
    for cone in cones:
        distances = mu.distance(cone, normalized_vertices)
        near_vertices = distances <= edge_dist
        elevations[near_vertices] += rng.uniform(-scale, scale, size=(np.sum(near_vertices),))
    return elevations


def _bfs_nearby_vertices(cores, adj_vertices, process_func=None, walk_range=5):
    queue = deque((v, 1) for v in cores)
    visited = set(cores)
    while queue:
        v, dis = queue.popleft()
        if process_func is not None:
            process_func(v, dis)
        if dis < walk_range:
            for u in adj_vertices[v]:
                if u == -1:
                    continue
                if u not in visited:
                    queue.append((u, dis + 1))
                    visited.add(u)


def cone_bfs_landscape(self, scale=1., n_cones=1, cone_range=5, percentile=None, decay_rate=5.):
    """Cone landscape, based on BFS.

    :param self:
    :param scale:
    :param n_cones:
    :param cone_range:
    :param percentile: int or 2-ints tuple
        Sample points from percentile (sorted by current elevations).
        None means sample from the whole map.
        Positive integers means sample from top percentile%.
        Negative integers means sample from bottom percentile%
        2-ints tuple means the range (both are positive integers).
    :param decay_rate:
    :return:
    """
    elevations = np.zeros_like(self.elevations)

    if percentile is None:
        candidates = self.n_vertices
    elif isinstance(percentile, int):
        if percentile > 0:
            op = operator.ge
        else:
            op = operator.le
            percentile = -percentile
        limit = np.percentile(self.elevations, percentile)
        candidates = [i for i, e in enumerate(self.elevations) if op(e, limit)]
    else:
        lo_percentile, hi_percentile = percentile
        lo_limit = np.percentile(self.elevations, lo_percentile)
        hi_limit = np.percentile(self.elevations, hi_percentile)
        candidates = [i for i, e in enumerate(self.elevations) if lo_limit <= e <= hi_limit]
    cores = np.random.choice(candidates, n_cones, replace=False)

    def _process_func(v, dis):
        elevation_add = mu.exp_2_dist((dis - 1) / cone_range, decay_rate=decay_rate, scale=scale)
        elevations[v] += elevation_add

    # Add elevations for island core and nearby vertices
    for island_core in cores:
        _bfs_nearby_vertices({island_core}, self.adj_vertices, process_func=_process_func, walk_range=cone_range)

    return elevations


def plateau_bfs_landscape(self, scale=1., plateau_range=25, slope_range=35):
    """Plateau landscape, based on BFS.

    :param self:
    :param scale:
    :param int plateau_range: BFS depth of plateau (elevation == scale)
    :param int slope_range: BFS depth of slope (elevation from scale to 0)
    :return:
    """
    elevations = np.zeros_like(self.elevations)
    core = np.random.choice(self.n_vertices)

    def _process_func(v, dis):
        if dis <= plateau_range:
            elevations[v] = scale
        else:
            elevations[v] = mu.exp_2_dist(
                (dis - 1) / (slope_range - plateau_range + 1), decay_rate=5., scale=scale)

    _bfs_nearby_vertices({core}, self.adj_vertices, process_func=_process_func, walk_range=slope_range)

    return elevations


def edge_penalty_landscape(self, scale=-1.):
    """Make sea in edge."""
    elevations = np.zeros_like(self.elevations)

    def _process_func(v, dis):
        elevations[v] = scale

    _bfs_nearby_vertices(set(self.adj_vertices[-1]), self.adj_vertices, process_func=_process_func, walk_range=2)
    return elevations


def smooth(elevations, iteration=1):
    """Smooth or sharp. Input elevations must be normalized to [0, 1]."""
    if iteration > 0:
        for _ in range(iteration):
            np.sqrt(elevations, out=elevations)
    else:
        for _ in range(-iteration):
            np.square(elevations, out=elevations)


def relax(self, elevations, iteration=1):
    """Replace each elevation value with the average of its neighbours."""
    for _ in range(iteration):
        new_elevations = np.zeros_like(elevations)
        for v, e in enumerate(elevations):
            new_elevations[v] = np.mean(elevations[self.adj_vertices[v]])
        elevations[:] = new_elevations


def clear_landscape(self, clear_edge_iter=1):
    """Clear landscape."""

    # 1. Remove edge lands: if an edge vertex is on land, and it has a sea neighbour, modify it to sea.
    for _ in range(clear_edge_iter):
        elevations = self.elevations
        for v, adj in self.adj_vertices.items():
            if v == -1:
                continue
            if elevations[v] < self.sea_level:
                continue
            if -1 not in adj:
                continue
            for u in adj:
                if u == -1:
                    continue
                if elevations[u] < self.sea_level:
                    elevations[v] = elevations[u]
                    break


def _set_landscape(self):
    self.elevations.fill(0)

    self.elevations += noise_landscape(self, scale=-0.01)

    for elevations in [
        # Some edge must be sea.
        cone_landscape(self, scale=-0.01, n_or_pos=[(0, 0.1 * i) for i in range(11)], decay_rate=25.),
        cone_landscape(self, scale=-0.01, n_or_pos=[(1, 0.1 * i) for i in range(11)], decay_rate=25.),
        cone_landscape(self, scale=-0.01, n_or_pos=[(0.1, 0), (0.2, 0)], decay_rate=25.),
        line_landscape(self, scale=-0.05, p1=(0.05, 0.09), p2=(0.07, 0.10), decay_rate=120.),
        slope_landscape(self, scale=0.5, slope=0.01, a=-1.0, b=-1.0, c=1.0),

        # North land
        plateau_landscape(self, scale=0.6, n_or_pos=[0.42, 0.91], edge_dist=0.02, decay_rate=60.),
        cone_landscape(self, scale=0.7, n_or_pos=[(0.50, 0.96)], decay_rate=70.),
        line_landscape(self, scale=0.3, p1=(0.34, 1.00), p2=(0.38, 0.91), decay_rate=200.),
        line_landscape(self, scale=0.2, p1=(0.31, 0.85), p2=(0.355, 0.695), decay_rate=200.),
        line_landscape(self, scale=0.4, p1=(0.49, 0.70), p2=(0.50, 0.65), decay_rate=800.),
        line_landscape(self, scale=0.4, p1=(0.51, 0.77), p2=(0.55, 0.72), decay_rate=800.),
        cone_landscape(self, scale=0.35, n_or_pos=[(0.62, 0.92)], decay_rate=1000.),
        cone_landscape(self, scale=0.20, n_or_pos=[(0.42, 0.99)], decay_rate=1000.),
        cone_landscape(self, scale=0.15, n_or_pos=[(0.40, 0.609)], decay_rate=1000.),

        # West land
        plateau_landscape(self, scale=0.2, n_or_pos=[(0.10, 0.80)], edge_dist=0.02, decay_rate=50.),
        cone_landscape(self, scale=0.6, n_or_pos=[(0.12, 0.19)], decay_rate=80.),
        plateau_landscape(self, scale=0.8, n_or_pos=[(0.17, 0.41)], edge_dist=0.12, decay_rate=40.),
        plateau_landscape(self, scale=0.4, n_or_pos=[(0.14, 0.58)], edge_dist=0.07, decay_rate=35.),
        line_landscape(self, scale=0.35, p1=(0.14, 0.78), p2=(0.20, 0.70), decay_rate=1200.),
        line_landscape(self, scale=0.35, p1=(0.24, 0.60), p2=(0.27, 0.61), decay_rate=1200.),
        cone_landscape(self, scale=0.7, n_or_pos=[(0.09, 0.85)], decay_rate=2700.),
        cone_landscape(self, scale=0.30, n_or_pos=[(0.30, 0.40)], decay_rate=1800.),
        cone_landscape(self, scale=0.10, n_or_pos=[(0.10, 0.69)], decay_rate=1000.),
        line_landscape(self, scale=-0.15, p1=(0.02, 0.39), p2=(0.21, 0.40), decay_rate=90.),
        plateau_landscape(self, scale=-0.25, n_or_pos=[(0.19, 0.42)], edge_dist=0.03, decay_rate=400.),
        line_landscape(self, scale=-0.15, p1=(0.19, 0.38), p2=(0.23, 0.36), decay_rate=500.),
        cone_landscape(self, scale=-0.20, n_or_pos=[(0.21, 0.39)], decay_rate=700.),

        # North-west landscape
        cone_landscape(self, scale=-0.2, n_or_pos=[(0.26, 0.71)], decay_rate=150.),
        line_landscape(self, scale=-0.2, p1=(0.29, 0.59), p2=(0.33, 0.55), decay_rate=200.),
        plateau_landscape(self, scale=-0.1, n_or_pos=[(0.15, 0.90)], edge_dist=0.02, decay_rate=80.),
        cone_landscape(self, scale=-0.08, n_or_pos=[(0.23, 0.91)], decay_rate=500.),
        cone_landscape(self, scale=0.2, n_or_pos=[(0.20, 0.87)], decay_rate=700.),
        cone_landscape(self, scale=0.1, n_or_pos=[(0.18, 0.79), (0.20, 0.78)], decay_rate=400.),
        line_landscape(self, scale=0.5, p1=(0.20, 0.82), p2=(0.25, 0.84), decay_rate=2000.),
        cone_landscape(self, scale=0.8, n_or_pos=[(0.14, 0.935)], decay_rate=4500.),
        cone_landscape(self, scale=0.18, n_or_pos=[(0.305, 0.60)], decay_rate=500.),
        cone_landscape(self, scale=0.26, n_or_pos=[(0.25, 0.67), (0.28, 0.71)], decay_rate=1300.),

        # North-east strait
        plateau_landscape(self, scale=-0.13, n_or_pos=[(0.70, 0.85)], edge_dist=0.02, decay_rate=700.),
        plateau_landscape(self, scale=-0.15, n_or_pos=[(0.67, 0.84)], edge_dist=0.02, decay_rate=700.),
        line_landscape(self, scale=-0.1, p1=(0.60, 0.80), p2=(0.68, 0.75), decay_rate=600.),

        # South-west strait
        cone_landscape(self, scale=-0.5, n_or_pos=[(0.16, 0.07)], decay_rate=1000.),
        cone_landscape(self, scale=-0.12, n_or_pos=[(0.34, 0.20)], decay_rate=1100.),
        line_landscape(self, scale=-0.1, p1=(0.32, 0.29), p2=(0.33, 0.35), decay_rate=200.),
        line_landscape(self, scale=-0.08, p1=(0.21, 0.20), p2=(0.23, 0.17), decay_rate=250.),
        cone_landscape(self, scale=0.20, n_or_pos=[(0.25, 0.16)], decay_rate=1800.),
        cone_landscape(self, scale=0.40, n_or_pos=[(0.30, 0.23)], decay_rate=3200.),

        # East land
        cone_landscape(self, scale=0.5, n_or_pos=[(0.73, 0.64), (0.74, 0.75)], decay_rate=80.),
        line_landscape(self, scale=0.4, p1=(0.59, 0.25), p2=(0.70, 0.35), decay_rate=120.),
        line_landscape(self, scale=0.1, p1=(0.75, 0.90), p2=(0.80, 0.86), decay_rate=800.),
        line_landscape(self, scale=0.1, p1=(0.62, 0.53), p2=(0.64, 0.49), decay_rate=800.),
        cone_landscape(self, scale=0.5, n_or_pos=[(0.65, 0.70)], decay_rate=1000.),
        cone_landscape(self, scale=0.2, n_or_pos=[(0.82, 0.50)], decay_rate=1000.),
        plateau_landscape(self, scale=0.2, n_or_pos=[(0.76, 0.29)], edge_dist=0.04, decay_rate=160.),
        cone_landscape(self, scale=0.09, n_or_pos=[(0.67, 0.47)], decay_rate=1000.),

        # South-east isthmus
        line_landscape(self, scale=-0.15, p1=(0.61, 0.29), p2=(0.66, 0.23), decay_rate=150.),
        cone_landscape(self, scale=-0.50, n_or_pos=[(0.63, 0.25)], decay_rate=1200.),

        # South land
        cone_landscape(self, scale=0.9, n_or_pos=[(0.42, 0.07)], decay_rate=80.),
        cone_landscape(self, scale=0.4, n_or_pos=[(0.32, 0.05)], decay_rate=200.),
        line_landscape(self, scale=0.4, p1=(0.39, 0.21), p2=(0.37, 0.30), decay_rate=600.),
        line_landscape(self, scale=0.05, p1=(0.70, 0.11), p2=(0.72, 0.12), decay_rate=500.),
        plateau_landscape(self, scale=0.4, n_or_pos=[(0.62, 0.03)], edge_dist=0.04, decay_rate=40.),
        plateau_landscape(self, scale=0.3, n_or_pos=[(0.52, 0.13)], edge_dist=0.12, decay_rate=50.),
        plateau_landscape(self, scale=0.2, n_or_pos=[(0.45, 0.31)], edge_dist=0.03, decay_rate=150.),
        plateau_landscape(self, scale=0.05, n_or_pos=[(0.56, 0.32)], edge_dist=0.01, decay_rate=500.),
        plateau_noise_landscape(self, scale=0.02, n_or_pos=[(0.57, 0.24)], edge_dist=0.03, seed=1),

        # Middle sea
        cone_landscape(self, scale=0.60, n_or_pos=[(0.34, 0.41)], decay_rate=6000.),
        cone_landscape(self, scale=0.30, n_or_pos=[(0.40, 0.49)], decay_rate=2500.),
        cone_landscape(self, scale=0.20, n_or_pos=[(0.48, 0.47)], decay_rate=2000.),
        cone_landscape(self, scale=0.10, n_or_pos=[(0.46, 0.50)], decay_rate=1900.),
        cone_landscape(self, scale=0.70, n_or_pos=[(0.57, 0.53)], decay_rate=2000.),
        line_landscape(self, scale=0.45, p1=(0.54, 0.42), p2=(0.55, 0.39), decay_rate=3000.),
        cone_landscape(self, scale=-0.05, n_or_pos=[(0.41, 0.65), (0.47, 0.67)], decay_rate=1000.),
        line_landscape(self, scale=-0.15, p1=(0.55, 0.38), p2=(0.60, 0.39), decay_rate=150.),
        cone_landscape(self, scale=-0.12, n_or_pos=[(0.60, 0.52)], decay_rate=1000.),
        cone_landscape(self, scale=-0.18, n_or_pos=[(0.55, 0.61)], decay_rate=2400.),
        cone_landscape(self, scale=-0.09, n_or_pos=[(0.46, 0.685)], decay_rate=1200.),
        plateau_landscape(self, scale=-0.10, n_or_pos=[(0.42, 0.42)], edge_dist=0.01, decay_rate=600.),

        # East ocean
        line_landscape(self, scale=-0.1, p1=(0.83, 0.70), p2=(0.85, 0.59), decay_rate=400.),
        line_landscape(self, scale=-0.2, p1=(0.79, 0.07), p2=(0.90, 0.12), decay_rate=600.),
        line_landscape(self, scale=-0.2, p1=(0.93, 0.15), p2=(1.00, 0.31), decay_rate=400.),
        line_landscape(self, scale=-0.05, p1=(0.80, 0.82), p2=(0.82, 0.81), decay_rate=500.),
        line_landscape(self, scale=-0.02, p1=(0.81, 0.25), p2=(0.85, 0.30), decay_rate=50.),
        cone_landscape(self, scale=-0.25, n_or_pos=[(0.84, 0.40)], decay_rate=600.),
        plateau_landscape(self, scale=-0.1, n_or_pos=[(0.90, 0.83)], edge_dist=0.05, decay_rate=800.),
        plateau_landscape(self, scale=-0.1, n_or_pos=[(0.71, 0.21)], edge_dist=0.01, decay_rate=100.),

        line_landscape(self, scale=0.4, p1=(0.84, 0.20), p2=(0.87, 0.22), decay_rate=3200.),
        cone_landscape(self, scale=0.18, n_or_pos=[(0.79, 0.19)], decay_rate=2400.),
        cone_landscape(self, scale=0.25, n_or_pos=[(0.93, 0.41)], decay_rate=2400.),
        plateau_landscape(self, scale=0.6, n_or_pos=[(0.95, 0.54)], edge_dist=0.02, decay_rate=2000.),
        cone_landscape(self, scale=0.3, n_or_pos=[(0.92, 0.63)], decay_rate=2800.),

        # Some landscape to fix too large lakes
        cone_landscape(self, scale=0.10, n_or_pos=[
            (0.277, 0.929), (0.170, 0.200),
            (0.391, 0.837), (0.412, 0.850), (0.346, 0.624),
            (0.428, 0.268), (0.440, 0.231),
            (0.660, 0.501),
        ], decay_rate=1400.),
    ]:
        self.elevations += elevations

    mu.normalize(self.elevations)
    relax(self, self.elevations, iteration=2)
    clear_landscape(self, clear_edge_iter=1)

    # % of sea
    self.sea_level = np.percentile(self.elevations, 50)


def _try_landscape(self):
    self.elevations.fill(0)

    mu.normalize(self.elevations)
    relax(self, self.elevations, iteration=2)
    clear_landscape(self, clear_edge_iter=1)

    # % of sea
    self.sea_level = np.percentile(self.elevations, self.args.sea_rate)


def set_landscape(self):
    _set_landscape(self)
    print('Set landscape done')
