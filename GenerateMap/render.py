#! /usr/bin/python
# -*- coding: utf-8 -*-

from queue import PriorityQueue

import numpy as np

try:
    from . import math_utils as mu
    from .erosion import get_downhill, get_water_flow
except ImportError:
    import math_utils as mu
    from erosion import get_downhill, get_water_flow


def _let_it_flow(self, downhill, start_points, infill=True):
    elevations = self.elevations
    vertices = self.vertices
    adj_vertices = self.adj_vertices
    normalized_vertices = mu.scale_pos_height(vertices, self)
    is_edge = self.is_edge
    start_points = np.asarray(start_points)
    mu.scale_point(start_points, self.width, self.height)
    start_vertices = {
        mu.nearest_vertex(p, normalized_vertices)
        for p in start_points}

    rivers = np.zeros_like(downhill) - 1
    lakes = np.zeros_like(downhill, dtype=np.bool)
    new_downhill = downhill.copy()

    def _build_lake(lake_start):
        lake_bottom_elevation = elevations[lake_start]
        lake_water_level = -10000
        visited = set()
        queue = PriorityQueue()
        queue.put((elevations[lake_start], lake_start))
        while queue:
            elevation_v, v = queue.get()
            if elevation_v > lake_water_level:
                lake_water_level = elevation_v
            if v in visited:
                continue
            visited.add(v)
            lakes[v] = True
            for u in adj_vertices[v]:
                elevation_u = elevations[u]
                if u not in visited:
                    if u == -1 or elevation_u < lake_bottom_elevation:
                        # v->u Lake output, all flow in the lake to here.
                        if infill:
                            for w in visited:
                                # Fill all vertices in the lake.
                                elevations[w] = lake_water_level + 1e-3 * (lake_water_level - elevations[w]) + 1e-5
                            # Erode the lake output.
                            elevations[v] = elevation_u + 1e-5
                        rivers[v] = u

                        # Change downhill for flow calculation.
                        for w in visited:
                            new_downhill[w] = v
                        new_downhill[v] = u
                        return u    # Lake output.
                    else:
                        queue.put((elevation_u, u))

    for start_vertex in start_vertices:
        v = start_vertex
        while True:
            if v == -1 or is_edge[v] or lakes[v] or rivers[v] != -1:
                break
            v_elevation = elevations[v]
            if v_elevation <= self.sea_level:
                break

            u_elevation, u = min((elevations[_u], _u) for _u in adj_vertices[v])
            if u_elevation < v_elevation:
                rivers[v] = u
                v = u
            else:
                lake_output = _build_lake(v)
                v = lake_output

    return rivers, lakes, new_downhill


def render_process(self):
    start_points = [
        # North
        (0.36, 0.916), (0.327, 0.848), (0.47, 0.94), (0.51, 0.91), (0.52, 0.94),
        (0.52, 0.75), (0.355, 0.689), (0.525, 0.775),
        # West
        (0.098, 0.805), (0.168, 0.755), (0.17, 0.46), (0.16, 0.56), (0.15, 0.61),
        (0.13, 0.60), (0.16, 0.34), (0.114, 0.185), (0.149, 0.405), (0.15, 0.501),
        (0.201, 0.465), (0.126, 0.214),
        # East
        (0.71, 0.61), (0.72, 0.36), (0.731, 0.691), (0.75, 0.60), (0.75, 0.78),
        (0.70, 0.355), (0.721, 0.591), (0.746, 0.651),
        # South
        (0.42, 0.10), (0.46, 0.12), (0.50, 0.14), (0.527, 0.100), (0.55, 0.15),
        (0.654, 0.055), (0.512, 0.172), (0.564, 0.109), (0.417, 0.092),
        (0.343, 0.059), (0.387, 0.091),
        # Middle
        (0.46, 0.495),
    ]

    flow_iteration = 9
    for i in range(flow_iteration):
        downhill = get_downhill(self, np.append(self.elevations, self.sea_level))
        _let_it_flow(self, downhill, start_points, infill=True)

    # Final calculate flow, not infill basins.
    downhill = get_downhill(self, np.append(self.elevations, self.sea_level))
    rivers, lakes, new_downhill = _let_it_flow(self, downhill, start_points, infill=False)

    # Recalculate water flow based on changed downhill.
    flow = get_water_flow(self, new_downhill, undersea=0.0)

    self.rivers = rivers
    self.lakes = lakes
    self.flow = flow


def auto_render_process(self):
    pass
