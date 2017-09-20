#! /usr/bin/python
# -*- coding: utf-8 -*-

from socket import socket, AF_INET, SOCK_STREAM

__author__ = 'fyabc'

ADDRESS = 'localhost', 20000
MSG_SIZE = 8192


def main():
    s = socket(AF_INET, SOCK_STREAM)
    s.connect(ADDRESS)

    while True:
        guess_str = input('Guess> ')
        s.sendall(guess_str.encode())
        msg = s.recv(MSG_SIZE).decode()

        print('Message from server: {}'.format(msg))

        if msg == 'Right!':
            s.close()
            break


if __name__ == '__main__':
    main()
