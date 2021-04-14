#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Server 1 (othello server using socket module)"""

import random
import socket
import threading
from multiprocessing.pool import ThreadPool
from typing import Dict

from constants import *
from network_utils import safe_send, safe_recv, repr_addr, safe_send_json, safe_recv_json


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


class ServerGameState:
    def __init__(self):
        self.players = {}   # type: Dict[int, PlayerInfo]
        self.current_player = 0
        self.observers = set()
        self.running = False
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
            'running': self.running,
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


def othello_server_thread(conn: socket.socket, addr: tuple, state: ServerGameState):
    _addr_r = repr_addr(addr)

    player_id = None
    ps = ''
    initialized = False

    while True:
        success, msg = safe_recv_json(conn)
        if not success:
            break

        msg_type = msg.get('type', None)

        if msg_type == 'init':
            player_id = state.new_player(msg['name'])

            ps = f'[Player {player_id}]'

            success = safe_send_json(conn, {
                'type': 'init',
                'player_id': player_id,
            })
            if not success:
                break

            initialized = True
        else:
            if not initialized:
                print(f'| {ps} not initialized')
                break

        if msg_type == 'quit':
            print(f'| {ps} quit game, auto lose')

            state.players[player_id].win_state = False
        else:
            print(f'[ERROR] | {ps} unknown message type')
            break

    print(f'| {ps} Lost connection')
    conn.close()


def threaded_client(conn: socket.socket, addr: tuple, state: ServerGameState):
    current_thread_name = threading.current_thread().name
    _addr_r = repr_addr(addr)

    # Send client ID
    running = safe_send(conn, current_thread_name)

    while running:
        success, data = safe_recv(conn)
        if not success:
            break

        if not data:
            print(f'| [{current_thread_name}] Disconnected from {_addr_r}')
            break

        print(f'| [{current_thread_name}] Received from {_addr_r}: {data}')
        print(f'| [{current_thread_name}]  Sending to {_addr_r}: {data}')

        success = safe_send(conn, data)
        if not success:
            break

    print(f'| [{current_thread_name}] Lost connection')
    conn.close()


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        sock.bind(SERVER_ADDR)
    except socket.error as e:
        print(f'[ERROR] | {e}')
        exit(1)

    sock.listen(MAX_CLIENTS)

    pool = ThreadPool(MAX_CLIENTS)

    print(f'| Server started at {repr_addr(SERVER_ADDR)}, waiting for a connection ...')

    game_state = ServerGameState()

    while True:
        conn, addr = sock.accept()
        print(f'| Connected to client {repr_addr(addr)}')

        pool.apply_async(threaded_client, (conn, addr, game_state))


if __name__ == '__main__':
    main()
