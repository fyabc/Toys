#! /usr/bin/python
# -*- coding: utf-8 -*-

import threading
import queue

from constants import *


class Client:
    def send_action(self, action):
        # TODO: Change this method into async version
        #  (the game should create a queue to receive server reply, and create a new thread to update states).
        pass


class LocalClient(Client):
    def __init__(self, local_server):
        super().__init__()
        self.server = local_server

    def send_action(self, action):
        return self.server.receive_action(action)


class SocketThreadingClient(Client):
    # TODO

    def __init__(self, server_address):
        super().__init__()
        self.server_address = server_address

    def send_action(self, action):
        pass
