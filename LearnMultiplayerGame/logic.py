#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Game logic."""

import random
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, asdict
from typing import List, Optional

from constants import *


@dataclass
class PlayerInfo:
    id: int
    ip_address: str
    name: str
    color: tuple
    status: str


class Board:
    def __init__(self, width, height):
        self._width = width
        self._height = height
        self.board = []     # type: List[List[Optional[int]]]

    @classmethod
    def from_board(cls, board: List[List[Optional[int]]], copy=True):
        width, height = len(board[0]), len(board)
        instance = cls(width, height)
        if copy:
            instance.board = deepcopy(board)
        else:
            instance.board = board
        return instance

    def reset(self, board=None, copy=True):
        if board is None:
            self.board = [[None for _ in range(self.width)] for _ in range(self.height)]
        else:
            if copy:
                self.board = deepcopy(board)
            else:
                self.board = board

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    def __getitem__(self, key):
        i, j = key  # type: int, int
        return self.board[i][j]

    def __setitem__(self, key, value):
        i, j = key  # type: int, int
        self.board[i][j] = value

    def n_empty(self):
        return sum(sum(p is None for p in row) for row in self.board)

    def count_players(self):
        counter = Counter()
        for row in self.board:
            counter.update(row)
        counter.pop(None)
        return counter
    
    def update(self, i: int, j: int, p: int):
        self[i, j] = p

        def _valid(_i, _j):
            return self[_i, _j] is not None and self[_i, _j] != p

        j1 = j - 1
        while j1 >= 0 and _valid(i, j1):
            j1 -= 1
        if j1 >= 0 and self[i, j1] == p:
            for jj in range(j1 + 1, j):
                self[i, jj] = p

        j1 = j + 1
        while j1 < self.width and _valid(i, j1):
            j1 += 1
        if j1 < self.width and self[i, j1] == p:
            for jj in range(j + 1, j1):
                self[i, jj] = p

        i1 = i - 1
        while i1 >= 0 and _valid(i1, j):
            i1 -= 1
        if i1 >= 0 and self[i1, j] == p:
            for ii in range(i1 + 1, i):
                self[ii, j] = p

        i1 = i + 1
        while i1 < self.height and _valid(i1, j):
            i1 += 1
        if i1 < self.height and self[i1, j] == p:
            for ii in range(i + 1, i1):
                self[ii, j] = p

        i2, j2 = i + 1, j + 1
        while i2 < self.height and j2 < self.width and _valid(i2, j2):
            i2 += 1
            j2 += 1
        if i2 < self.height and j2 < self.width and self[i2, j2] == p:
            for ii, jj in zip(range(i + 1, i2), range(j + 1, j2)):
                self[ii, jj] = p

        i2, j2 = i - 1, j - 1
        while i2 >= 0 and j2 >= 0 and _valid(i2, j2):
            i2 -= 1
            j2 -= 1
        if i2 >= 0 and j2 >= 0 and self[i2, j2] == p:
            for ii, jj in zip(range(i2 + 1, i), range(j2 + 1, j)):
                self[ii, jj] = p

        i2, j2 = i + 1, j - 1
        while i2 < self.height and j2 >= 0 and _valid(i2, j2):
            i2 += 1
            j2 -= 1
        if i2 < self.height and j2 >= 0 and self[i2, j2] == p:
            for ii, jj in zip(range(i + 1, i2), range(j - 1, j2, -1)):
                self[ii, jj] = p

        i2, j2 = i - 1, j + 1
        while i2 >= 0 and j2 < self.width and _valid(i2, j2):
            i2 -= 1
            j2 += 1
        if i2 >= 0 and j2 < self.width and self[i2, j2] == p:
            for ii, jj in zip(range(i - 1, i2, -1), range(j + 1, j2)):
                self[ii, jj] = p

        # TODO: Diagonals.

    def iter_board(self):
        for i in range(self.height):
            for j in range(self.width):
                yield i, j, self[i, j]


class GameState:
    def __init__(self):
        self.board = Board(CONF.G.BOARD_WIDTH, CONF.G.BOARD_HEIGHT)     # type: Board
        self.current_player_id = 0
        self.players = [None for _ in range(CONF.G.MAX_PLAYERS)]   # type: list[Optional[PlayerInfo]]
        self.audiences = []

        self.current_pos = None    # type: Optional[tuple]

        self.reset()

    def reset(self):
        self.board.reset()
        self.players = [None for _ in range(CONF.G.MAX_PLAYERS)]
        self.audiences.clear()
        self.current_pos = None

    def state_dict(self):
        return {
            'current_player_id': self.current_player_id,
            'board': self.board.board,
            'players': [None if info is None else asdict(info) for info in self.players],
            'current_pos': self.current_pos,
        }

    def load_state_dict(self, state_dict: dict):
        self.current_player_id = state_dict['current_player_id']
        self.board.reset(state_dict['board'])
        self.players = [None if info_state_dict is None else PlayerInfo(**info_state_dict)
                        for info_state_dict in state_dict['players']]
        self.current_pos = state_dict['current_pos']


class ServerGameState(GameState):
    def server_game_start(self):
        n_players = len(self.players)
        assert n_players == 2
        self.current_player_id = 0

        i2 = CONF.G.BOARD_WIDTH // 2
        j2 = CONF.G.BOARD_HEIGHT // 2
        self.board[i2 - 1, j2 - 1] = self.board[i2, j2] = 0
        self.board[i2, j2 - 1] = self.board[i2 - 1, j2] = 1

    def _next_player(self):
        self.current_player_id = (self.current_player_id + 1) % len(self.players)

    def _update_board(self, i, j, player_id):
        self.current_pos = (i, j)
        self.board.update(i, j, player_id)

        # Count pieces.
        piece_counts = self.board.count_players()
        n_empty = self.board.n_empty()
        if n_empty == 0:
            pass

    def take_action(self, action: dict) -> dict:
        a_type = action['type']

        if a_type == 'init':
            available_ids = [i for i, p in enumerate(self.players) if p is None]
            player_id = random.choice(available_ids)
            player = PlayerInfo(
                id=player_id,
                ip_address=action['ip_address'],
                name=action['name'],
                color=CONF.U.PLAYER_COLORS[player_id],
                status='playing',
            )
            self.players[player_id] = player
            if all(self.players):
                self.server_game_start()
        elif a_type == 'play':
            self._update_board(action['i'], action['j'], action['player_id'])
            self._next_player()
        else:
            raise RuntimeError(f'Unknown action type {a_type}')
        return self.state_dict()
