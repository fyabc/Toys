#! /usr/bin/python
# -*- coding: utf-8 -*-

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

WIDTH_PIXELS = 1200
MAX_RIVER_WIDTH = 2.2

try:
    DivergingNorm = colors.DivergingNorm
except AttributeError:
    class DivergingNorm(colors.Normalize):
        def __init__(self, vcenter, vmin=None, vmax=None):
            """
            Normalize data with a set center.

            Useful when mapping data with an unequal rates of change around a
            conceptual center, e.g., data that range from -2 to 4, with 0 as
            the midpoint.

            Parameters
            ----------
            vcenter : float
                The data value that defines ``0.5`` in the normalization.
            vmin : float, optional
                The data value that defines ``0.0`` in the normalization.
                Defaults to the min value of the dataset.
            vmax : float, optional
                The data value that defines ``1.0`` in the normalization.
                Defaults to the the max value of the dataset.

            Examples
            --------
            This maps data value -4000 to 0., 0 to 0.5, and +10000 to 1.0; data
            between is linearly interpolated::

                >>> import matplotlib.colors as mcolors
                >>> offset = mcolors.DivergingNorm(vmin=-4000.,
                                                   vcenter=0., vmax=10000)
                >>> data = [-4000., -2000., 0., 2500., 5000., 7500., 10000.]
                >>> offset(data)
                array([0., 0.25, 0.5, 0.625, 0.75, 0.875, 1.0])
            """

            self.vcenter = vcenter
            self.vmin = vmin
            self.vmax = vmax
            if vcenter is not None and vmax is not None and vcenter >= vmax:
                raise ValueError('vmin, vcenter, and vmax must be in '
                                 'ascending order')
            if vcenter is not None and vmin is not None and vcenter <= vmin:
                raise ValueError('vmin, vcenter, and vmax must be in '
                                 'ascending order')

        def autoscale_None(self, A):
            """
            Get vmin and vmax, and then clip at vcenter
            """
            super().autoscale_None(A)
            if self.vmin > self.vcenter:
                self.vmin = self.vcenter
            if self.vmax < self.vcenter:
                self.vmax = self.vcenter

        def __call__(self, value, clip=None):
            """
            Map value to the interval [0, 1]. The clip argument is unused.
            """
            result, is_scalar = self.process_value(value)
            self.autoscale_None(result)  # sets self.vmin, self.vmax if None

            if not self.vmin <= self.vcenter <= self.vmax:
                raise ValueError("vmin, vcenter, vmax must increase monotonically")
            result = np.ma.masked_array(
                np.interp(result, [self.vmin, self.vcenter, self.vmax],
                          [0, 0.5, 1.]), mask=np.ma.getmask(result))
            if is_scalar:
                result = np.atleast_1d(result)[0]
            return result


def get_color_factory(lo=0, hi=1, sea_level=0.5):
    colors_undersea = plt.cm.terrain(np.linspace(0, 0.17, 256))
    colors_land = plt.cm.terrain(np.linspace(0.25, 1, 256))
    all_colors = np.vstack((colors_undersea, colors_land))
    terrain_map = colors.LinearSegmentedColormap.from_list('terrain_map', all_colors)

    div_norm = DivergingNorm(vcenter=sea_level, vmin=lo, vmax=hi)

    def _elevation2color(elevation):
        return terrain_map(div_norm(elevation))
    return _elevation2color


def _plot_surface(self):
    vertex_regions = self.vertex_regions
    elevations = self.elevations
    points = self.points
    sea_level = self.sea_level
    color_factory = get_color_factory(sea_level=sea_level)

    for v in range(self.n_vertices):
        regions = vertex_regions[v]
        polygon = points[regions]
        elevation = elevations[v]
        color = color_factory(elevation)
        plt.fill(polygon[:, 0], polygon[:, 1], color=color)


def _plot_coastline(self):
    vertex_regions = self.vertex_regions
    elevations = self.elevations
    points = self.points
    sea_level = self.sea_level

    for v, adj in self.adj_vertices.items():
        if v == -1:
            continue
        for u in adj:
            if u <= v:
                continue
            if (elevations[u] > sea_level) ^ (elevations[v] > sea_level):
                adj_points = list(set(vertex_regions[u]) & set(vertex_regions[v]))
                x1, y1 = points[adj_points[0]]
                x2, y2 = points[adj_points[1]]
                plt.plot([x1, x2], [y1, y2], 'k-', linewidth=1.0)


def _plot_rivers(self, smooth_rivers=True):
    vertices = self.vertices
    max_flow = np.max(self.flow)

    # # Debug plot.
    # showed = set()
    # visited = has_river_indicator | self.lakes
    # for v in np.arange(self.n_vertices)[visited]:
    #     x1, y1 = vertices[v]
    #     if v not in showed:
    #         showed.add(v)
    #         plt.text(x1, y1, 'E{:.4f}/F{:.3f}‰'.format(self.elevations[v], self.flow[v] * 1000))
    #     for u in self.adj_vertices[v]:
    #         if u == -1:
    #             continue
    #         x2, y2 = vertices[u]
    #         if u not in showed:
    #             showed.add(u)
    #             plt.text(x2, y2, 'E{:.4f}/F{:.3f}‰'.format(self.elevations[u], self.flow[u] * 1000))
    #         plt.plot([x1, x2], [y1, y2], 'r-', linewidth=1)
    # lake_vertices = vertices[self.lakes]
    # plt.scatter(lake_vertices[:, 0], lake_vertices[:, 1], color='g')

    if smooth_rivers:
        # Smooth rivers: use midpoints instead of original vertices.
        rivers = self.rivers
        flow = self.flow
        river_downhill_pos = vertices[rivers]
        river_midpoint_pos = (vertices + river_downhill_pos) / 2
        for v in range(self.n_vertices):
            u = rivers[v]
            if u == -1 or rivers[u] == -1:
                continue
            x1, y1 = river_midpoint_pos[v]
            x2, y2 = river_midpoint_pos[u]
            f = flow[v]
            line_width = MAX_RIVER_WIDTH * np.sqrt(f / max_flow)
            plt.plot([x1, x2], [y1, y2], 'b-', linewidth=line_width)
    else:
        has_river_indicator = self.rivers != -1
        river_vertices = np.arange(self.n_vertices)[has_river_indicator]
        river_vertex_pos = vertices[river_vertices]
        river_downhill_vertices = self.rivers[has_river_indicator]
        river_downhill_pos = vertices[river_downhill_vertices]
        river_flows = self.flow[has_river_indicator]
        for p_v, p_u, f in zip(river_vertex_pos, river_downhill_pos, river_flows):
            x1, y1 = p_v
            x2, y2 = p_u
            # line_width = 2.5 * f / max_flow
            line_width = MAX_RIVER_WIDTH * np.sqrt(f / max_flow)
            plt.plot([x1, x2], [y1, y2], 'b-', linewidth=line_width)

    points = self.points
    lakes = self.lakes
    for v in np.arange(self.n_vertices)[lakes]:
        regions = self.vertex_regions[v]
        polygon = points[regions]
        plt.fill(polygon[:, 0], polygon[:, 1], color='b')


def _plot_auto_rivers(self, smooth_rivers=True):
    pass


def _plot_cities(self):
    capital_pos = self.vertices[self.capitals]
    city_pos = self.vertices[self.cities]

    plt.scatter(capital_pos[:, 0], capital_pos[:, 1], s=24, facecolors='w', edgecolors='k', zorder=3)
    plt.scatter(city_pos[:, 0], city_pos[:, 1], s=15, facecolors='k', edgecolors='none', zorder=3)


def plot_topographic_map(self, surface=True, river=True, cities=True):
    if surface:
        _plot_surface(self)

    if self.auto_render:
        _plot_auto_rivers(self)
    else:
        _plot_rivers(self)

    _plot_coastline(self)

    if cities and hasattr(self, 'capitals'):
        _plot_cities(self)

    # TODO: Add height bar.
    show_fig('Elevation map', self.width, self.height)


def show_fig(title: str, width, height):
    plt.xlim(xmin=0, xmax=width)
    plt.ylim(ymin=0, ymax=height)
    plt.xticks(np.linspace(0, width, num=11), [format(i / 10, '.1f') for i in range(11)])
    plt.yticks(np.linspace(0, height, num=11), [format(i / 10, '.1f') for i in range(11)])

    plt.title(title)
    plt.grid()

    gcf = plt.gcf()
    _width_inches = WIDTH_PIXELS / gcf.dpi
    gcf.set_size_inches(_width_inches, _width_inches * height / width)

    gcf.set_tight_layout(True)
    plt.show()
