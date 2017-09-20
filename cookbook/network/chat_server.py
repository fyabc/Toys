#! /usr/bin/python
# -*- coding: utf-8 -*-

import re
import socket
from socketserver import StreamRequestHandler, ThreadingTCPServer

__author__ = 'fyabc'

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


class ClientError(Exception):
    """An exception thrown because the client gave bad input to the server."""
    pass


class ChatServer(ThreadingTCPServer):
    """The server class."""

    def __init__(self, server_address, RequestHandlerClass, bind_and_activate=True):
        super().__init__(server_address, RequestHandlerClass, bind_and_activate)

        # Key: user name, Value: output fd
        self.users = {}


class ChatHandler(StreamRequestHandler):
    """Handles the life cycle of a user's connection to the chat server: connecting, chatting,
    running server commands, and disconnecting.
    """

    NICKNAME = re.compile('^[A-Za-z0-9_-]+$')  # regex for a valid nickname

    def __init__(self, request, client_address, server):
        self.nickname = None
        self.commands = {
            'users': self.c_users,
            'nick': self.c_nick,
            'quit': self.c_quit,
            'help': self.c_help,
            '?': self.c_help,
        }

        super().__init__(request, client_address, server)

    def setup(self):
        super().setup()

    def handle(self):
        """Handles a connection: gets the user's nickname, then processes input from the user
        until they quit or drop the connection.
        """

        self.nickname = None
        self.private_msg('Who are you?')
        nickname = self.read_line()

        done = False
        try:
            self.c_nick(nickname)
            self.private_msg('Hello `{}`, welcome to the Python chatter {}!'.format(
                nickname, self.server.server_address))
            self.broadcast('{} has joined into the chat.'.format(nickname), False)
            print('{} has joined into the chat.'.format(nickname))
        except ClientError as e:
            self.private_msg(e.args[0])
            done = True
        except socket.error:
            done = True

        while not done:
            try:
                done = self.parse_input()
            except ClientError as e:
                self.private_msg(str(e))
            except socket.error:
                done = True

    def finish(self):
        if self.nickname:
            # The user successfully connected before disconnecting.
            # Broadcast that they're quitting to everyone else.
            message = '{} has quit.'.format(self.nickname)
            if hasattr(self, 'parting_words'):
                message = '{} has quit: {}'.format(self.nickname, self.parting_words)
            self.broadcast(message, False)
            print(message)

            # Remove the user from the list so we don't keep trying
            # to send them messages.
            if self.server.users.get(self.nickname):
                del (self.server.users[self.nickname])

        super().finish()

    # Commands.

    def c_nick(self, nickname):
        """Attempts to change a user's nickname."""

        if not nickname:
            raise ClientError('No nickname provided.')
        if not self.NICKNAME.match(nickname):
            raise ClientError('Invalid nickname: {}'.format(nickname))
        if nickname == self.nickname:
            raise ClientError('You\'re already known as {}.'.format(nickname))
        if self.server.users.get(nickname, None):
            raise ClientError('There\'s already a user named "{}" here.'.format(nickname))

        oldNickname = None
        if self.nickname:
            oldNickname = self.nickname
            del (self.server.users[self.nickname])
        self.server.users[nickname] = self.wfile
        self.nickname = nickname

        if oldNickname:
            self.broadcast('{} is now known as {}'.format(oldNickname, self.nickname))

        return False

    def c_quit(self, parting_words):
        """Tells the other users that this user has quit, then makes
        sure the handler will close this connection."""

        if parting_words:
            self.parting_words = parting_words
        return True

    def c_users(self, _):
        """Returns a list of the users in this chat room."""

        self.private_msg('Chat room users: ' + ', '.join(self.server.users.keys()))

        return False

    def c_help(self, command):
        self.private_msg('Chat System Help')
        # todo
        return False

    def parse_input(self):
        """Reads a line from the socket input and either runs it as a command, or broadcasts it as chat text."""

        done = False
        msg = self.read_line()

        command, arg = self._parse_command(msg)

        if command:
            done = command(arg)
        else:
            msg = '<{}> {}'.format(self.nickname, msg)
            self.broadcast(msg)
        return done

    # Helper methods.

    def _parse_command(self, msg):
        """Try to parse a string as a command to the server.
        If it's an implemented command, run the corresponding method.
        """

        command_method, arg = None, None
        if msg and msg[0] == '/':
            if len(msg) < 2:
                raise ClientError('Invalid command: "{}"'.format(msg))
            command_and_arg = msg[1:].split(' ', 1)
            if len(command_and_arg) == 2:
                command, arg = command_and_arg
            else:
                command = command_and_arg[0]
            command_method = self.commands.get(command, None)
            if not command_method:
                raise ClientError('No such command: "{}"'.format(command))

        # if input[0] != '/', which means input is not a command
        # then command_method will be None
        return command_method, arg

    def broadcast(self, msg, include_this_user=True):
        """Send a message to every connected user, possibly exempting the user who's the cause of the message."""

        for user, f_out in self.server.users.items():
            if user == self.nickname and not include_this_user:
                continue
            _send(f_out, msg)

    def private_msg(self, msg):
        _send(self.wfile, msg)

    def read_line(self):
        return _recv(self.rfile)


def main():
    server = ChatServer(ADDRESS, ChatHandler)
    server.serve_forever()


if __name__ == '__main__':
    main()
