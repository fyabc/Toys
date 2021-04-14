#! /usr/bin/python
# -*- coding: utf-8 -*-

import pygame

_FONT_PATH_CACHE = {}
_FONT_CACHE = {}


def get_font(font_name: str, font_size: int) -> pygame.font.Font:
    font = _FONT_CACHE.get((font_name, font_size), None)
    if font is None:
        font_path = _FONT_PATH_CACHE.get(font_name, None)
        if font_path is None:
            font_path = _FONT_PATH_CACHE[font_name] = pygame.font.match_font(font_name)
        font = _FONT_CACHE[(font_name, font_size)] = pygame.font.Font(font_path, font_size)
    return font


def reverse_color(color: tuple) -> tuple:
    r, g, b, a = color
    return 255 - r, 255 - g, 255 - b, a
