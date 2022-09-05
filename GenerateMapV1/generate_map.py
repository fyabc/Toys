#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Generate fantasy map.

See http://mewo2.com/notes/terrain/.
"""

import argparse
import time
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial as spl

try:
    from . import erosion
    from . import plot_utils as pu
    from . import proto_landscapes as pl
    from . import render
    from . import political
except ImportError:
    import erosion
    import proto_landscapes as pl
    import render
    import plot_utils as pu
    import political


WIDTH = 177
HEIGHT = 100


class MapGrid:
    def __init__(self, args):
        self.args = args
        self.width, self.height = args.width, args.height
        self.n = args.grid_points
        self.auto_render = args.auto_render
        self._debug_grid = args.debug_grid
        self._debug_landscape = args.debug_landscape

        self._build_grid()

        self._build_elevation_map()

        self._do_erosion()

        self._render_terrain()

        self._render_political()

    def _build_grid(self):
        self.points = np.zeros((self.n, 2))
        self.points[:, 0] = np.random.uniform(0, self.width, (self.n,))
        self.points[:, 1] = np.random.uniform(0, self.height, (self.n,))
        if self._debug_grid:
            plt.scatter(self.points[:, 0], self.points[:, 1])
            pu.show_fig('Points', self.width, self.height)

        self._improve_points()

        self.vor = spl.Voronoi(self.points)

        # Regions, represented by their center points.
        self.regions = [self.vor.regions[i] for i in self.vor.point_region]

        # Vertices of Voronoi polygons (except -1).
        # All calculations will on vertices (not points)
        self.vertices = self.vor.vertices
        self.n_vertices = self.vertices.shape[0]

        if self._debug_grid:
            plt.scatter(self.vertices[:, 0], self.vertices[:, 1])
            plt.triplot(self.points[:, 0], self.points[:, 1], color='k')
            pu.show_fig('Voronoi vertices', self.width, self.height)

        self._build_adj()

        self._improve_vertices()

    def _improve_points(self, iteration=2):
        for it in range(iteration):
            vor = spl.Voronoi(self.points)

            new_points = []

            for i in range(len(vor.points)):
                point = vor.points[i, :]
                region = vor.regions[vor.point_region[i]]
                if -1 in region:
                    new_points.append(point)
                else:
                    vertices = np.asarray([vor.vertices[j, :] for j in region])
                    vertices[vertices < 0] = 0
                    vertices[(vertices[:, 0] > self.width), 0] = self.width
                    vertices[(vertices[:, 1] > self.height), 1] = self.height
                    new_point = np.mean(vertices, 0)
                    new_points.append(new_point)

            self.points = np.asarray(new_points)
            if self._debug_grid:
                plt.scatter(self.points[:, 0], self.points[:, 1])
                pu.show_fig('Points after improvement iteration {}'.format(it + 1), self.width, self.height)

    def _build_adj(self):
        """Build adjacent tables of the Voronoi."""
        self.adj_points = defaultdict(list)
        for p1, p2 in self.vor.ridge_points:
            self.adj_points[p1].append(p2)
            self.adj_points[p2].append(p1)

        self.adj_vertices = defaultdict(list)
        for v1, v2 in self.vor.ridge_vertices:
            if v2 != -1 or -1 not in self.adj_vertices[v1]:
                self.adj_vertices[v1].append(v2)
            if v1 != -1 or -1 not in self.adj_vertices[v2]:
                self.adj_vertices[v2].append(v1)

        # Each vertex will have exact 3 neighbours in Voronoi.
        self.adj_mat = np.zeros((self.n_vertices, 3), np.int32) - 1
        for v, adj in self.adj_vertices.items():
            if v != -1:
                if len(adj) < 3:
                    adj.extend(-1 for _ in range(3 - len(adj)))
                self.adj_mat[v, :] = adj

        # Vertices (except -1) and their adjacent regions.
        self.vertex_regions = defaultdict(list)
        # Like vertex_regions, but include -1.
        self.tris = defaultdict(list)
        for p, region in enumerate(self.regions):
            for v in region:
                self.tris[v].append(p)
                if v != -1:
                    self.vertex_regions[v].append(p)

        # Edge (infinite) vertices or not.
        self.is_edge = np.zeros(self.n_vertices, np.bool)
        for v, adj in self.adj_vertices.items():
            if v == -1:
                continue
            if -1 in adj:
                self.is_edge[v] = True

    def _improve_vertices(self):
        """Improve vertices to avoid too far from the map."""
        for v in range(self.n_vertices):
            self.vertices[v, :] = np.mean(self.points[self.vertex_regions[v]], 0)
        if self._debug_grid:
            plt.scatter(self.vertices[:, 0], self.vertices[:, 1])
            plt.triplot(self.points[:, 0], self.points[:, 1], color='k')
            pu.show_fig('Voronoi regions after vertices improvement', self.width, self.height)

    def _build_elevation_map(self):
        # Elevations of all vertices
        self.elevations = np.zeros(self.n_vertices)
        self.sea_level = 0.5

        pl.set_landscape(self)

        self.erosivity = np.ones((self.n_vertices,))
        erosion.set_erosivity(self)

        if self._debug_landscape:
            pu.plot_topographic_map(self, surface=self.args.debug_plot_surface)

    def _do_erosion(self):
        erosion.erosion_process(self)
        print('Erosion done')
        if self.args.debug_erosion:
            pu.plot_topographic_map(self, surface=self.args.debug_plot_surface)

    def _render_terrain(self):
        if self.auto_render:
            render.auto_render_process(self)
        else:
            render.render_process(self)
        print('Render done')
        if self.args.debug_render:
            pu.plot_topographic_map(self, surface=self.args.debug_plot_surface)

    def _render_political(self):
        political.place_cities(self, n_capitals=self.args.n_capitals, n_cities=self.args.n_cities)
        print('Political render done')
        if self.args.debug_political:
            pu.plot_topographic_map(self, surface=self.args.debug_plot_surface)


def main(args=None):
    parser = argparse.ArgumentParser('Generate fantasy map.')
    parser.add_argument('-s', '--seed', type=int, default=1, help='Random seed, default is %(default)r.')
    parser.add_argument('-W', '--width', type=float, default=WIDTH, help='Width, default is %(default)r.')
    parser.add_argument('-H', '--height', type=float, default=HEIGHT, help='Height, default is %(default)r.')
    parser.add_argument('-N', '--grid-points', type=int, default=1 << 8,
                        help='Number of grid points, default is %(default)r.')
    parser.add_argument('--sea-rate', type=float, default=50., help='Sea rate, default is %(default)r.')
    parser.add_argument('-E', '--num-erosion-iter', type=int, default=10,
                        help='Number of erosion iteration, default is %(default)r.')
    parser.add_argument('--rain-rate', type=float, default=1.0, help='Rain rate, default is %(default)r')
    parser.add_argument('--erode-rate', type=float, default=0.10, help='Erode rate, default is %(default)r')
    parser.add_argument('--undersea-erode-cut', type=float, default=0.25,
                        help='Cut erode rate undersea, default is %(default)r')
    parser.add_argument('--ar', '--auto-render', dest='auto_render', action='store_true', default=False,
                        help='Render rivers automatically')
    parser.add_argument('--n-capitals', type=int, default=15, help='Number of capitals, default is %(default)r')
    parser.add_argument('--n-cities', type=int, default=20, help='Number of common cities, default is %(default)r')
    parser.add_argument('-DG', '--debug-grid', action='store_true', default=False,
                        help='Show detailed grid build process.')
    parser.add_argument('-DL', '--debug-landscape', action='store_true', default=False,
                        help='Show detailed landscape build process.')
    parser.add_argument('-DE', '--debug-erosion', action='store_true', default=False,
                        help='Show detailed erosion process.')
    parser.add_argument('-DR', '--debug-render', action='store_true', default=False,
                        help='Show detailed render process.')
    parser.add_argument('-DS', '--debug-no-plot-surface', dest='debug_plot_surface', action='store_false', default=True,
                        help='Not plot surface in debug')
    parser.add_argument('-DP', '--debug-political', action='store_true', default=False,
                        help='Show detail political render process.')

    args = parser.parse_args(args=args)

    np.random.seed(args.seed)

    start_time = time.time()
    map_grid = MapGrid(args)
    print('Time passed: {:.3f}s'.format(time.time() - start_time))


if __name__ == '__main__':
    # height map time:
    # N     | Surface | No surface | No Surface |
    #       |         |            | + River    |
    # 2048  | 24s     | 3.6s       |
    # 3072  | 40s     | 4.4s
    # 4096  | 63s     | 6s
    # 6144  | 124s    | 6.9s
    # 8192  | 217s    | 7.5s       |            |
    # 16384 | 830s    | 12s
    # 32768 | 2722s   | 21s        | 50s        |
    # 65536 |         | 40s
    # 2048: 24s; 3072: 40s; 4096: 63s; 6144: 124s; 8192: 217s; 16384: 830s; 32768: 2722s;
    main('-N 32768 -s 1 -DP -DS'.split())
