#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Pygame drawing methods."""

import pygame

from constants import *


def init() -> pygame.Surface:
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Client')

    return win


def redraw_window(win: pygame.Surface, world):
    world.draw(win)

    pygame.display.update()


def finalize():
    pygame.quit()


def main_loop(game):
    win = init()
    clock = pygame.time.Clock()

    running = True
    while running:
        dt = clock.tick(FPS)

        for event in pygame.event.get():    # type: pygame.event.EventType
            if event.type == pygame.QUIT:
                running = False

        game.update(dt)

        redraw_window(win, game.world)

    finalize()
