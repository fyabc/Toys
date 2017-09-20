#! /usr/bin/python
# -*- coding: utf-8 -*-

from socketserver import BaseRequestHandler, TCPServer, ThreadingTCPServer

__author__ = 'fyabc'

ADDRESS = 'localhost', 20000
MSG_SIZE = 8192


class EchoHandler(BaseRequestHandler):
    def handle(self):
        print('Get connection from', self.client_address)
        # todo
        while True:
            msg = self.request.recv(MSG_SIZE)
            if not msg:
                break
            print('Message (type {}): {}'.format(type(msg), msg))
            self.request.sendall(msg)


def main():
    TCPServer.allow_reuse_address = True
    server = TCPServer(ADDRESS, EchoHandler)
    server.serve_forever()


if __name__ == '__main__':
    main()
