#! /usr/bin/python
# -*- coding: utf-8 -*-

from socket import socket, AF_INET, SOCK_STREAM

__author__ = 'fyabc'

ADDRESS = 'localhost', 20000
MSG_SIZE = 8192


def main():
    s = socket(AF_INET, SOCK_STREAM)
    s.connect(ADDRESS)
    s.sendall(b'Hello')

    print(s.recv(MSG_SIZE))


if __name__ == '__main__':
    main()
