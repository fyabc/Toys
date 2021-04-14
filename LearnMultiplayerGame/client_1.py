#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Client 1 (client using socket module)"""

import socket

from constants import *
from network_utils import safe_send, safe_recv, repr_addr


class Client:
    def __init__(self):
        self.running = True
        try:
            self.client = socket.create_connection(SERVER_ADDR)
        except socket.error as e:
            self.running = False
            print(f'[ERROR] | {e}')
            return
        print(f'| Connected to server {repr_addr(SERVER_ADDR)}')

        self.running, self.id = safe_recv(self.client)
        if not self.running:
            return

    def send(self, data: str):
        send_success = safe_send(self.client, data)
        if not send_success:
            return ''
        recv_success, reply = safe_recv(self.client)
        return reply


def main():
    client = Client()
    print(f'| Client ID: {client.id}')

    while True:
        import time
        time.sleep(1.0)

        reply = client.send(f'Hello from client {client.id}')
        print(reply)


if __name__ == '__main__':
    main()
