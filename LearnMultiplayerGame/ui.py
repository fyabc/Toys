#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Game UI design."""

import random
from typing import List

import pygame
from pygame.color import THECOLORS

from constants import *


class MySprite:
    def __init__(self):
        self.children = []  # type: List[MySprite]

    def add_child(self, child: 'MySprite'):
        self.children.append(child)
        return child

    def draw(self, win: pygame.Surface):
        stop = self._draw(win)
        for child in self.children:
            if not stop:
                stop = child.draw(win)
        return stop

    def _draw(self, win: pygame.Surface):
        pass

    def update(self, dt: float):
        stop = self._update(dt)
        for child in self.children:
            if not stop:
                stop = child.update(dt)
        return stop

    def _update(self, dt: float):
        pass


class Player(MySprite):
    def __init__(self, x=0., y=0., width=50., height=50., color=THECOLORS['black']):
        super().__init__()

        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.velocity = 0.5

    @classmethod
    def random_pos(cls, color):
        x = random.uniform(0, WIDTH)
        y = random.uniform(0, HEIGHT)
        width = 50
        height = 50
        return cls(x, y, width, height, color)

    @property
    def rect(self):
        return self.x, self.y, self.width, self.height

    def _draw(self, win: pygame.Surface):
        pygame.draw.rect(win, self.color, self.rect)

    def _update(self, dt: float):
        keys = pygame.key.get_pressed()

        delta = self.velocity * dt
        if keys[pygame.K_LEFT]:
            self.x -= delta
        if keys[pygame.K_RIGHT]:
            self.x += delta
        if keys[pygame.K_UP]:
            self.y -= delta
        if keys[pygame.K_DOWN]:
            self.y += delta


class World(MySprite):
    def __init__(self, state):
        super().__init__()
        self.state = state

    def _draw(self, win: pygame.Surface):
        win.fill(THECOLORS['white'])

        # Board
        for j in range(BOARD_SIZE[0] + 1):
            x_pos = BOARD_EDGES['left'] + j * (BOARD_EDGES['right'] - BOARD_EDGES['left']) / BOARD_SIZE[0]
            pygame.draw.line(
                win, BOARD_EDGE_COLOR,
                (x_pos, BOARD_EDGES['top']),
                (x_pos, BOARD_EDGES['bottom']),
                2,
            )
        for i in range(BOARD_SIZE[1] + 1):
            y_pos = BOARD_EDGES['top'] + i * (BOARD_EDGES['bottom'] - BOARD_EDGES['top']) / BOARD_SIZE[1]
            pygame.draw.line(
                win, BOARD_EDGE_COLOR,
                (BOARD_EDGES['left'], y_pos),
                (BOARD_EDGES['right'], y_pos),
                2,
            )
