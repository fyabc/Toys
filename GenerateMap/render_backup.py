#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Backup old render method, from the original Python repository."""

import matplotlib.pyplot as plt
import numpy as np

try:
    from .erosion import get_downhill, get_water_flow
except ImportError:
    from erosion import _get_downhill, _get_water_flow


def _get_sinks(self, downhill):
    sinks = downhill.copy()
    water = self.elevations <= self.sea_level
    sink_list = np.where((sinks == -1) & ~water & ~self.is_edge)[0]
    sinks[sink_list] = sink_list
    sinks[water] = -1

    while True:
        new_sinks = sinks.copy()
        new_sinks[~water] = sinks[sinks[~water]]
        new_sinks[sinks == -1] = -1
        if np.all(new_sinks == sinks):
            break
        sinks = new_sinks
    return sinks


def _find_lowest_sill(self, sinks):
    h = 10000
    edges = np.where(
        (sinks != -1) &
        np.any((sinks[self.adj_mat] == -1) & self.adj_mat != -1, 1)
    )[0]

    best_uv = None, None
    for u in edges:
        for v in self.adj_vertices[u]:
            if v == -1:
                continue
            if sinks[v] == -1:
                new_h = max(self.elevations[v], self.elevations[u])
                if new_h < h:
                    h = new_h
                    best_uv = u, v
    assert h < 10000
    u, v = best_uv
    return h, u, v


def _infill(self):
    """Infill the sink vertices (lower than all neighbours)."""

    elevations = self.elevations
    downhill = self.downhill
    for _ in range(1, 11):
        print('#', _)
        sinks = _get_sinks(self, downhill)
        if np.all(sinks == -1):
            return
        h, u, v = _find_lowest_sill(self, sinks)
        sink = sinks[u]
        if downhill[v] != -1:
            elevations[v] = self.elevations[downhill[v]] + 1e-5
        sink_elevations = self.elevations[sinks == sink]
        h = np.where(sink_elevations < h, h + 0.001 * (h - sink_elevations), sink_elevations) + 1e-5
        self.elevations[sinks == sink] = h
        self.downhill = get_downhill(self, np.append(self.elevations, self.sea_level))


def render_process_old(self):
    ext_elevations = np.append(self.elevations, self.sea_level)
    self.downhill = get_downhill(self, ext_elevations)
    _infill(self)
    self.flow = get_water_flow(self, self.downhill, undersea=0.0)


def _plot_rivers_old(self):
    river_percentile = 93
    vertices = self.vertices
    downhill = self.downhill
    flow = self.flow
    flow_threshold = np.percentile(flow, river_percentile)
    flow_max = np.max(flow)
    river_indicator = flow > flow_threshold
    river_indicator &= (downhill != -1)     # Only use valid downhill.

    river_vertices = np.arange(self.n_vertices)[river_indicator]
    downhill_vertices = downhill[river_indicator]

    river_from_points = vertices[river_vertices]
    river_to_points = vertices[downhill_vertices]
    river_flows = flow[river_indicator]
    for (x1, y1), (x2, y2), river_flow in zip(river_from_points, river_to_points, river_flows):
        flow_width = 2.0 * (river_flow - flow_threshold) / (flow_max - flow_threshold)
        plt.plot([x1, x2], [y1, y2], 'b-', linewidth=flow_width)
