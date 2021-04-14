#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Game logic."""

import random
import threading
from typing import Dict

from constants import *


class PlayerInfo:
    def __init__(self, player_id):
        self.id = player_id
        self.name = ''
        self.win_state = None
        self.color = PLAYER_COLORS[self.id]

    def state_dict(self):
        return {
            'id': self.id,
            'win_state': self.win_state,
            'name': self.name,
            'color': self.color,
        }


class GameState:
    def __init__(self):
        self.players = {}  # type: Dict[int, PlayerInfo]
        self.current_player = 0
        self.observers = set()
        self.state = 'waiting'
        self.board = [[None for _ in range(BOARD_WIDTH)] for _ in range(BOARD_HEIGHT)]
        self.lock = threading.Lock()

    def reset(self):
        pass

    def new_player(self, name):
        with self.lock:
            available_player = set(range(MAX_PLAYERS)) - self.players.keys()
            assert available_player, 'No available player'
            player_id = random.choice(list(available_player))
            self.players[player_id] = PlayerInfo(player_id)
            self.players[player_id].name = name
            return player_id

    def state_dict(self):
        return {
            'players': {
                player_id: info.state_dict()
                for player_id, info in self.players.items()
            },
            'current_player': self.current_player,
            'state': self.state,
            'board': self.board,
        }

    def update(self, player_id, i, j):
        with self.lock:
            self.board[i][j] = player_id

        # TODO: Update by othello rule.

    def check_winner(self):
        n_empty_cells = sum(sum(item is None for item in row) for row in self.board)
        if n_empty_cells == 0:
            pass
