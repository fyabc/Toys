#! /usr/bin/python
# -*- coding: utf-8 -*-

__author__ = 'fyabc'

import socket
import select
import sys
import os
from threading import Thread

ADDRESS = 'localhost', 20000
MSG_SIZE = 8192


def _ensure_newline(s):
    if s and s[-1] != '\n':
        s += '\r\n'
    return s


def _send(fd, msg):
    fd.write(_ensure_newline(msg).encode())


def _recv(fd):
    return fd.readline().strip().decode()


class ChatClient:
    def __init__(self, address):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect(address)
        self.input = self.socket.makefile('rb', 0)
        self.output = self.socket.makefile('wb', 0)

        # Send the given nickname to the server.
        authentication_demand = _recv(self.input)
        if not authentication_demand.startswith("Who are you?"):
            raise Exception("This doesn't seem to be a Python Chat Server.")
        print(authentication_demand)

        _send(self.output, input() + '\r\n')
        response = _recv(self.input)

        if not response.startswith("Hello"):
            raise Exception(response)
        print(response)

        # Start out by printing out the list of members.
        _send(self.output, '/users\r\n')
        print(_recv(self.input))

        self.run()

    def run(self):
        """Start a separate thread to gather the input from the
        keyboard even as we wait for messages to come over the
        network. This makes it possible for the user to simultaneously
        send and receive chat text."""

        propagate_standard_input = self.PropagateStandardInput(self.output)
        propagate_standard_input.start()

        # Read from the network and print everything received to standard
        # output. Once data stops coming in from the network, it means
        # we've disconnected.
        input_text = True
        while input_text:
            try:
                input_text = _recv(self.input)
            except ConnectionError:
                break
            if input_text:
                print(input_text.strip())
        propagate_standard_input.done = True

    class PropagateStandardInput(Thread):
        """A class that mirrors standard input to the chat server
        until it's told to stop."""

        def __init__(self, output):
            """Make this thread a daemon thread, so that if the Python
            interpreter needs to quit it won't be held up waiting for this
            thread to die."""
            Thread.__init__(self)
            self.setDaemon(True)
            self.output = output
            self.done = False

        def run(self):
            """Echo standard input to the chat server until told to stop."""

            while not self.done:
                input_text = input()
                if input_text:
                    _send(self.output, input_text + '\r\n')


def main():
    ChatClient(ADDRESS)


if __name__ == '__main__':
    main()
