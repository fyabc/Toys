#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Simple test script to run my server and client."""

import sys
import time

import server
import client
from game_state import GameState
import message

SERVER_ADDR = 'localhost', 5555


def msg_dummy(client_name):
    return message.Message(
        type='dummy',
        data={'message': f'dummy server broadcast by client {client_name}'},
    )


class EchoGameState(GameState):
    def __init__(self):
        super().__init__()
        self.clients = {}

    def update_server(self, msg: 'message.Message') -> 'message.Message':
        assert msg.type == 'client_init'
        self.clients[msg.client_address] = msg['user_name']

        client_name = self.clients[msg.client_address]
        print(f'Server received message from client {client_name}')
        return msg_dummy(client_name)

    def update_client(self, resp):
        pass


class EchoClient(client.StatefulGameClient):
    def __init__(self, state):
        super().__init__(SERVER_ADDR, state)
        self.register_observer(lambda msg: print(f'Client {self.client_address} received {msg}'))


def main():
    state = EchoGameState()
    if sys.argv[1].lower() in {'server', 's'}:
        my_server = server.spawn_server(SERVER_ADDR, state)
    else:
        assert sys.argv[1].lower() in {'client', 'c'}

        my_client = EchoClient(state)

        while my_client.running:
            my_client.send_message(message.msg_client_init(sys.argv[2]))

            time.sleep(1.0)


if __name__ == '__main__':
    main()
