#! /usr/bin/python
# -*- coding: utf-8 -*-

"""My general client framework."""

import threading
import socket

from message import Message
import network_utils as nu


class GameClient:
    def __init__(self, server_address):
        self.server_address = server_address
        self.observers = []

        self._running = threading.Event()

        try:
            self.socket = socket.create_connection(server_address)
        except ConnectionError as e:
            print(f'| [Client] {e}')
            print(f'| [Client] cannot connect to server {self.server_address}')
            return
        else:
            print(f'| [Client] connected to server {self.server_address}')

        self.client_address = self.socket.getsockname()

        self.receiver_thread = threading.Thread(target=self._receiver_main)
        self._running.set()
        self.receiver_thread.start()

    def send_message(self, msg: Message):
        if not self.running:
            return

        msg_bytes = msg.to_bytes()
        try:
            nu.send_message(self.socket, msg_bytes)
        except ConnectionError as e:
            print(e)
            self.shutdown()

    def register_observer(self, observer):
        self.observers.append(observer)

    @property
    def running(self):
        return self._running.is_set()

    def _receiver_main(self):
        while self.running:
            try:
                resp_data = nu.recv_message(self.socket)
            except ConnectionError as e:
                print(f'| [Client] {e}')
                break

            resp = Message.from_bytes(resp_data)
            for observer in self.observers:
                observer(resp)

        print(f'| [Client] client receiver stopped')

    def shutdown(self):
        self._running.clear()
        self.socket.close()
        print(f'| [Client] disconnected from server {self.server_address}')


class StatefulGameClient(GameClient):
    def __init__(self, server_address, state):
        super().__init__(server_address)
        self.state = state

        self.register_observer(self.update_client_state)

    def update_client_state(self, resp):
        self.state.update_client(resp)
