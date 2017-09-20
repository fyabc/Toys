#! /usr/bin/python
# -*- coding: utf-8 -*-

import pickle
from socket import socket, AF_INET, SOCK_STREAM

__author__ = 'fyabc'

ADDRESS = 'localhost', 20000
MSG_SIZE = 8192


def send_msg(soc, msg):
    soc.sendall(msg.encode())


def recv_msg(soc):
    return soc.recv(MSG_SIZE).decode()


def send_object(soc, obj):
    soc.sendall(pickle.dumps(obj))


def recv_object(soc):
    return pickle.loads(soc.recv(MSG_SIZE))


def main():
    s = socket(AF_INET, SOCK_STREAM)
    s.connect(ADDRESS)

    send_msg(s, 'Search for an opponent')

    # Wait for opponent
    while True:
        # Blocked here. How to change it into unblocked?
        msg = recv_msg(s)

        print('$', msg)

        if msg == 'Game Start!':
            break

    # Game
    # todo


if __name__ == '__main__':
    main()
