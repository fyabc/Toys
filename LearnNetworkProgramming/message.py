#! /usr/bin/python
# -*- coding: utf-8 -*-

import json
import time


class Message:
    def __init__(self, type: str, data: dict, timestamp: float = None):
        self.timestamp = time.time() if timestamp is None else timestamp
        self.type = type
        self.data = data
        self.client_address = None

    def set_client_address(self, client_address):
        self.client_address = client_address

    def to_bytes(self) -> bytes:
        return json.dumps({
            'type': self.type,
            'timestamp': self.timestamp,
            'data': self.data,
        }).encode('UTF-8')

    @classmethod
    def from_bytes(cls, data: bytes):
        json_msg = json.loads(data.decode('UTF-8'))
        return cls(
            type=json_msg['type'],
            data=json_msg['data'],
            timestamp=json_msg['timestamp'],
        )

    def __getitem__(self, item):
        return self.data[item]

    def __repr__(self):
        return f'Message({self.__dict__})'


def msg_client_init(user_name: str, **kwargs):
    return Message(
        type='client_init',
        data={
            'user_name': user_name,
            **kwargs,
        },
    )


def msg_server_resp(state):
    return Message(
        type='server_response',
        data=state.state_dict(),
    )
