#! /usr/bin/python
# -*- coding: utf-8 -*-

from message import Message


class GameState:
    def update_server(self, msg) -> 'Message':
        pass

    def update_client(self, resp):
        pass

    def snapshot(self):
        pass

    def state_dict(self) -> dict:
        raise NotImplementedError()

    def upgrade_state_dict(self, state_dict: dict):
        raise NotImplementedError()
