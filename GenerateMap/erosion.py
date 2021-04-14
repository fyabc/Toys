#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Do the erosion."""

from collections import deque

import numpy as np
import scipy.sparse as spa
import scipy.sparse.linalg as sla

try:
    from . import math_utils as mu
except ImportError:
    import math_utils as mu


def get_downhill(self, ext_elevations):
    """Get "downhill" neighbour of each vertex."""
    lowest_neighbour_indices = np.argmin(ext_elevations[self.adj_mat], 1)
    downhill = self.adj_mat[np.arange(self.n_vertices), lowest_neighbour_indices]

    # If lower than all neighbours or on edge, set downhill to -1.
    downhill[self.elevations <= ext_elevations[downhill]] = -1
    downhill[self.is_edge] = -1

    return downhill


def _get_rain(self):
    n = self.n_vertices

    rain = np.ones(n)
    rain /= np.sum(rain)
    rain *= self.args.rain_rate

    return rain


def get_water_flow(self, downhill, undersea=0.0):
    n = self.n_vertices

    rain = _get_rain(self)

    has_downhill_indices = downhill != -1
    row = downhill[has_downhill_indices]
    col = np.arange(n)[has_downhill_indices]

    # Flow = self rain + all flow from uphill.
    # rain + down_matrix * flow = flow
    # => (I - down_matrix) * flow = rain, solve the linear system.
    d_mat = spa.eye(n) - spa.coo_matrix((np.ones_like(row), (row, col)), shape=(n, n)).tocsc()
    flow = sla.spsolve(d_mat, rain)

    # Cut undersea
    flow[self.elevations <= self.sea_level] *= undersea

    return flow


def _get_slopes(self, downhill, ext_elevations):
    dist = mu.all_distances(self.vertices, self.vertices[downhill, :])
    slope = (self.elevations - ext_elevations[downhill]) / (dist + 1e-9)
    slope[downhill == -1] = 0
    return slope


def _erode(self, flow, slope, undersea=0.0):
    erode_rate = self.args.erode_rate

    river_rate = -flow ** 0.5 * slope  # River erosion
    slope_rate = -slope ** 2 * self.erosivity  # Slope smoothing
    rate = 800 * river_rate + 2 * slope_rate
    rate[self.elevations <= self.sea_level] *= undersea  # Cut undersea
    self.elevations += rate / np.abs(rate).max() * erode_rate


def _clean_coastline(self, iteration=3, outwards=True, clean_inner_sea=True):
    sea_level = self.sea_level
    ext_elevations = np.append(self.elevations, sea_level)
    for _ in range(iteration):
        new_elevations = ext_elevations[:-1].copy()

        # Clean islands.
        for v in range(self.n_vertices):
            if self.is_edge[v] or ext_elevations[v] <= sea_level:
                continue
            adj = self.adj_vertices[v]
            adj_elevations = ext_elevations[adj]
            if np.sum(adj_elevations > sea_level) <= 1:
                new_elevations[v] = np.mean(adj_elevations[adj_elevations <= sea_level])
        ext_elevations[:-1] = new_elevations

        if outwards:
            # Clean lakes.
            for v in range(self.n_vertices):
                if self.is_edge[v] or ext_elevations[v] > sea_level:
                    continue
                adj = self.adj_vertices[v]
                adj_elevations = ext_elevations[adj]
                if np.sum(adj_elevations <= sea_level) <= 1:
                    new_elevations[v] = np.mean(adj_elevations[adj_elevations > sea_level])
            ext_elevations[:-1] = new_elevations

    self.elevations = ext_elevations[:-1]

    if clean_inner_sea:
        # Clean all inner sea.
        elevations = self.elevations
        adj_vertices = self.adj_vertices
        root_ocean_pos = (0.0, 0.0)
        root_ocean_vertex = mu.nearest_vertex(root_ocean_pos, self.vertices)
        assert elevations[root_ocean_vertex] <= sea_level
        queue = deque([root_ocean_vertex])
        visited = np.zeros_like(self.elevations, dtype=np.bool)
        visited[root_ocean_vertex] = True
        while queue:
            v = queue.popleft()
            for u in adj_vertices[v]:
                if u == -1 or visited[u] or elevations[u] > sea_level:
                    continue
                queue.append(u)
                visited[u] = True
        elevations[~visited & (elevations <= sea_level)] = sea_level + 1e-5


def set_erosivity(self):
    pass


def erosion_process(self):
    set_erosivity(self)

    for _ in range(self.args.num_erosion_iter):
        # Extended elevations, append sea level (for -1)
        ext_elevations = np.append(self.elevations, self.sea_level)
        downhill = get_downhill(self, ext_elevations)
        flow = get_water_flow(self, downhill, undersea=self.args.undersea_erode_cut)
        slope = _get_slopes(self, downhill, ext_elevations)
        _erode(self, flow, slope, undersea=self.args.undersea_erode_cut)

        mu.normalize(self.elevations)
        self.sea_level = np.percentile(self.elevations, self.args.sea_rate)

    _clean_coastline(self, iteration=2)
