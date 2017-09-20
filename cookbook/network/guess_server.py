#! /usr/bin/python
# -*- coding: utf-8 -*-

import random
from socketserver import BaseRequestHandler, TCPServer

__author__ = 'fyabc'

ADDRESS = 'localhost', 20000
MSG_SIZE = 8192


class GuessHandler(BaseRequestHandler):
    def __init__(self, request, client_address, server):
        self.number = random.randint(0, 100)
        print('My secret number is: {}'.format(self.number))
        super().__init__(request, client_address, server)

    def handle(self):
        print('Get connection from', self.client_address)
        while True:
            try:
                msg = self.request.recv(MSG_SIZE)
            except ConnectionAbortedError:
                print('Connection closed by client.')
                break

            msg_str = msg.decode()

            print('Message from connection: {}'.format(msg_str))

            try:
                i = int(msg_str)
            except ValueError:
                self.request.sendall('Not a valid integer!'.encode())
                continue

            if i < self.number:
                self.request.sendall('Too small!'.encode())
            elif i > self.number:
                self.request.sendall('Too large!'.encode())
            else:
                self.request.sendall('Right!'.encode())


def main():
    server = TCPServer(ADDRESS, GuessHandler)
    server.serve_forever()


if __name__ == '__main__':
    main()
