#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Game logic."""

import random
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


class GameState:
    def __init__(self):
        self.board = []   # type: List[List[Optional[int]]]
        self.current_player_id = 0
        self.players = [None for _ in range(CONF.G.MAX_PLAYERS)]   # type: list[Optional[PlayerInfo]]
        self.audiences = []

        self.current_pos = None    # type: Optional[tuple]

        self.reset()

    def reset(self):
        self.board = [[None for _ in range(CONF.G.BOARD_WIDTH)] for _ in range(CONF.G.BOARD_HEIGHT)]
        self.players = [None for _ in range(CONF.G.MAX_PLAYERS)]
        self.audiences.clear()
        self.current_pos = None

    def state_dict(self):
        return {
            'current_player_id': self.current_player_id,
            'board': self.board,
            'players': [None if info is None else asdict(info) for info in self.players],
            'current_pos': self.current_pos,
        }

    def load_state_dict(self, state_dict: dict):
        self.current_player_id = state_dict['current_player_id']
        self.board = deepcopy(state_dict['board'])
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
        self.board[i2 - 1][j2 - 1] = self.board[i2][j2] = 0
        self.board[i2][j2 - 1] = self.board[i2 - 1][j2] = 1

    def _next_player(self):
        self.current_player_id = (self.current_player_id + 1) % len(self.players)

    def _update_board(self, i, j, player_id):
        self.current_pos = (i, j)
        self.board[i][j] = player_id

        j1 = j - 1
        while j1 >= 0 and self.board[i][j1] is not None and self.board[i][j1] != player_id:
            j1 -= 1
        if j1 >= 0 and self.board[i][j1] == player_id:
            for jj in range(j1 + 1, j):
                self.board[i][jj] = player_id

        j1 = j + 1
        while j1 < CONF.G.BOARD_WIDTH and self.board[i][j1] is not None and self.board[i][j1] != player_id:
            j1 += 1
        if j1 < CONF.G.BOARD_WIDTH and self.board[i][j1] == player_id:
            for jj in range(j + 1, j1):
                self.board[i][jj] = player_id

        i1 = i - 1
        while i1 >= 0 and self.board[i1][j] is not None and self.board[i1][j] != player_id:
            i1 -= 1
        if i1 >= 0 and self.board[i1][j] == player_id:
            for ii in range(i1 + 1, i):
                self.board[ii][j] = player_id

        i1 = i + 1
        while i1 < CONF.G.BOARD_HEIGHT and self.board[i1][j] is not None and self.board[i1][j] != player_id:
            i1 += 1
        if i1 < CONF.G.BOARD_HEIGHT and self.board[i1][j] == player_id:
            for ii in range(i + 1, i1):
                self.board[ii][j] = player_id

        # Count pieces.
        piece_counts = [0 for _ in self.players]
        for row in self.board:
            for cell in row:
                if cell is not None:
                    piece_counts[cell] += 1
        if sum(piece_counts) == CONF.G.BOARD_WIDTH * CONF.G.BOARD_HEIGHT:
            pass
        # TODO: Diagonals.

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
