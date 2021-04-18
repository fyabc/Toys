#! /usr/bin/python
# -*- coding: utf-8 -*-

import socket
import struct


MSG_HEADER_FMT = '>I'


def send_message(conn: socket.socket, msg: bytes):
    msg = struct.pack(MSG_HEADER_FMT, len(msg)) + msg
    conn.sendall(msg)


def recv_message(conn: socket.socket):
    header = _recv_all(conn, 4)
    if not header:
        return b''
    msg_len = struct.unpack(MSG_HEADER_FMT, header)[0]
    return _recv_all(conn, msg_len)


def _recv_all(conn: socket.socket, n: int):
    data = bytearray()
    while len(data) < n:
        packet = conn.recv(min(n - len(data), 1024))
        if not packet:
            return b''
        data.extend(packet)
    return data
