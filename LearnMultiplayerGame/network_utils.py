#! /usr/bin/python
# -*- coding: utf-8 -*-

import socket
import json

from constants import *


def repr_addr(addr: tuple):
    host, port = addr
    return f'{host}:{port}'


def _safe_send(conn: socket.socket, data: bytes) -> bool:
    try:
        conn.sendall(data)
    except socket.error as e:
        print(f'[ERROR] | {e}')
        return False
    return True


def safe_send(conn: socket.socket, data: str) -> bool:
    return _safe_send(conn, data.encode(ENCODING))


def safe_send_json(conn: socket.socket, data) -> bool:
    data = json.dumps(data)
    data = data.encode(ENCODING)
    return _safe_send(conn, data)


def _safe_recv(conn: socket.socket) -> (bool, bytes):
    try:
        data = conn.recv(BUF_SIZE)
    except socket.error as e:
        print(f'[ERROR] | {e}')
        return False, b''
    return True, data


def safe_recv(conn: socket.socket) -> (bool, str):
    success, data = _safe_recv(conn)
    return success, data.decode(ENCODING)


def safe_recv_json(conn: socket.socket):
    success, data = _safe_recv(conn)
    data = data.decode(ENCODING)
    data = json.loads(data)
    return success, data
