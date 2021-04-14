#! /usr/bin/python
# -*- coding: utf-8 -*-


class Client:
    def send_action(self, action):
        pass


class LocalClient(Client):
    def __init__(self, local_server):
        super().__init__()
        self.server = local_server

    def send_action(self, action):
        return self.server.receive_action(action)


class SocketClient(Client):
    def send_action(self, action):
        pass
