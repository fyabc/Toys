#! /usr/bin/python
# -*- coding: utf-8 -*-

"""My general server framework."""

import socketserver
import threading

from message import Message
import network_utils as nu


class GameMessageHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        server = self.server    # type: GameServer

        server.add_client(self.request, self.client_address)
        print(f'| [Server] connected to client {self.client_address}')

        while True:
            request = self.request

            try:
                data = nu.recv_message(request)
            except ConnectionError as e:
                print(f'| [Server] {e}')
                break

            if not data:
                # Reach EOF.
                break

            msg = Message.from_bytes(data)
            msg.set_client_address(self.client_address)

            response = server.update_state(msg)
            resp_data = response.to_bytes()

            server.broadcast(resp_data)

        print(f'| [Server] client {self.client_address} disconnected')
        server.remove_client(self.client_address)


class GameServer(socketserver.ThreadingTCPServer):
    def __init__(self, address, state):
        super().__init__(address, GameMessageHandler)
        self._state = state
        self._clients = {}
        self._lock = threading.Lock()

    def add_client(self, client_socket, client_address):
        self._clients[client_address] = client_socket

    def remove_client(self, client_address):
        del self._clients[client_address]

    def broadcast(self, msg_data: bytes):
        for socket in self._clients.values():
            try:
                nu.send_message(socket, msg_data)
            except ConnectionError as e:
                print(f'| [Server] {e}')
                continue

    def update_state(self, msg) -> 'Message':
        with self._lock:
            return self._state.update_server(msg)

    def serve_forever(self, poll_interval: float = 0.5) -> None:
        print('| [Server] starting serve ...')
        return super().serve_forever(poll_interval)


# TODO: Add support of thread pool server.


def spawn_server(address, state, return_thread=False, poll_interval=0.5):
    server = GameServer(address, state)

    server_thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=poll_interval))
    server_thread.start()

    if return_thread:
        return server, server_thread
    else:
        return server
