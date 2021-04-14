#! /usr/bin/python
# -*- coding: utf-8 -*-

from pygame.color import THECOLORS

SERVER_HOST = '192.168.31.30'
SERVER_PORT = 5555
SERVER_ADDR = (SERVER_HOST, SERVER_PORT)
MAX_CLIENTS = 2
MAX_PLAYERS = 2
BUF_SIZE = 2048
ENCODING = 'UTF-8'

WIDTH, HEIGHT = 960, 640
FPS = 30
PLAYER_COLORS = {
    0: THECOLORS['black'],
    1: THECOLORS['white'],
    2: THECOLORS['blue'],
    3: THECOLORS['red'],
}
BOARD_EDGES = {
    'left': 40,
    'right': 600,
    'top': 40,
    'bottom': 600,
}
BOARD_EDGE_COLOR = THECOLORS['black']

BOARD_WIDTH = 8
BOARD_HEIGHT = 8
BOARD_SIZE = (BOARD_WIDTH, BOARD_HEIGHT)
