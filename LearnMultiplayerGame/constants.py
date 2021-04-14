#! /usr/bin/python
# -*- coding: utf-8 -*-

from pygame.color import THECOLORS


class _SN:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


CONF = _SN()

# Network
CONF.N = _SN()
CONF.N.SERVER_HOST = '192.168.31.30'
CONF.N.SERVER_PORT = 5555

# UI
CONF.U = _SN()
CONF.U.WIDTH = 960
CONF.U.HEIGHT = 640
CONF.U.FPS = 45
CONF.U.PLAYER_COLORS = {
    0: THECOLORS['black'],
    1: THECOLORS['white'],
    2: THECOLORS['blue'],
    3: THECOLORS['red'],
}
CONF.U.BOARD_EDGES = {
    'left': 40,
    'right': 600,
    'top': 40,
    'bottom': 600,
}
CONF.U.BOARD_LINE_WIDTH = 2
CONF.U.BOARD_EDGE_COLOR = THECOLORS['black']
CONF.U.FONT_NAME = None
CONF.U.FONT_SIZE = 48
CONF.U.PLAYER_LABEL_POS = [
    (780, 120),
    (780, 200),
]

# Game
CONF.G = _SN()
CONF.G.BOARD_WIDTH = 8
CONF.G.BOARD_HEIGHT = 8
CONF.G.BOARD_SIZE = (CONF.G.BOARD_WIDTH, CONF.G.BOARD_HEIGHT)
CONF.G.MAX_PLAYERS = 2
CONF.G.MAX_AUDIENCES = 10

SERVER_HOST = '192.168.31.30'
SERVER_PORT = 5555
SERVER_ADDR = (SERVER_HOST, SERVER_PORT)
MAX_CLIENTS = 2
BUF_SIZE = 2048
ENCODING = 'UTF-8'
