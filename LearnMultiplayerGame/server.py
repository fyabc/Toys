#! /usr/bin/python
# -*- coding: utf-8 -*-

import threading
from queue import Queue

from logic import ServerGameState


class LocalServer:
    def __init__(self):
        self.state = ServerGameState()

    def receive_action(self, action: dict) -> dict:
        reply = self.state.take_action(action)
        return reply


class LocalThreadingServer:
    pass
