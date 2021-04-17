#! /usr/bin/python
# -*- coding: utf-8 -*-

import logging
import threading
from queue import Queue

from logic import ServerGameState


class Server:
    def __init__(self):
        self.state = ServerGameState()

    def receive_action(self, action: dict) -> dict:
        raise NotImplementedError()

    def stop(self):
        pass


class LocalServer(Server):
    def __init__(self):
        super().__init__()

    def receive_action(self, action: dict) -> dict:
        reply = self.state.take_action(action)
        return reply


class LocalThreadingServer(Server):
    _TERMINATE_THREAD_EVENT = object()

    def __init__(self):
        super().__init__()
        self.in_queue = Queue()
        self.out_queue = Queue()

        self.running = threading.Event()
        self.running.set()
        self._t = threading.Thread(target=self._run)
        self._t.start()

    def _run(self):
        while self.running.is_set():
            action = self.in_queue.get()
            if action == self._TERMINATE_THREAD_EVENT:
                break
            reply = self.state.take_action(action)
            self.out_queue.put(reply)

    def receive_action(self, action: dict) -> dict:
        if not self.running:
            logging.critical('server not running')
            return self.state.state_dict()

        self.in_queue.put(action)
        server_reply = self.out_queue.get()
        return server_reply

    def stop(self):
        self.running.clear()
        self.in_queue.put(self._TERMINATE_THREAD_EVENT)


class SocketThreadingServer(Server):
    def __init__(self):
        super().__init__()
