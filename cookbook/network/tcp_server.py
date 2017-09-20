#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Server of Rock Paper Scissor game (2 players)."""

from socketserver import BaseRequestHandler, TCPServer

__author__ = 'fyabc'

ADDRESS = 'localhost', 20000
MSG_SIZE = 8192


class RpsHandler(BaseRequestHandler):
    def handle(self):
        print('Get connection from', self.client_address)
        while True:
            msg = self.request.recv(MSG_SIZE)
            if not msg:
                break
            print('Message from {}: {}'.format(self.client_address, msg))
            self.request.sendall(msg)


def main():
    server = TCPServer(ADDRESS, RpsHandler)
    server.serve_forever()


if __name__ == '__main__':
    main()
