#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np

try:
    from . import math_utils as mu
except ImportError:
    import math_utils as mu


def place_cities(self, n_capitals, n_cities):
    city_score = self.flow ** 0.5
    city_score[self.elevations <= self.sea_level] = -1e9

    if not self.auto_render:
        city_score[self.rivers != -1] = -1e9
        city_score[self.lakes] = -1e9
    else:
        # TODO
        pass

    vertices = self.vertices
    normalized_vertices = mu.scale_pos_height(vertices, self)

    city_score[vertices[:, 0] <= 0.05 * self.width] = -1e9
    city_score[vertices[:, 0] >= 0.95 * self.width] = -1e9
    city_score[vertices[:, 1] <= 0.05 * self.height] = -1e9
    city_score[vertices[:, 1] >= 0.95 * self.height] = -1e9

    self.capitals = []
    self.cities = []
    for _n, _l in zip([n_capitals, n_cities], [self.capitals, self.cities]):
        while len(_l) < _n:
            new_city = np.argmax(city_score)

            _l.append(new_city)
            city_score -= 0.01 * 1 / (mu.distance(normalized_vertices[new_city], normalized_vertices) + 1e-9)
