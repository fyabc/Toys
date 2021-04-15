#! /usr/bin/python
# -*- coding: utf-8 -*-

import math
from typing import List

import pygame

import logic
import utils
import server
import client
from framework import Widget, Label, ReverseColorButton
from constants import *


class ChessPiece(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.center = kwargs.get('center', self.center)
        self.color = kwargs.get('color', THECOLORS['black'])
        self.border_width = kwargs.get('border_width', 2)
        self.border_color = kwargs.get('border_color', THECOLORS['black'])
        self.callback = kwargs.get('callback', None)

        if self.width != self.height:
            raise ValueError('width != height for chess piece')

    @property
    def radius(self):
        return self.width / 2

    def collide_point(self, x, y):
        center = self.center
        return math.hypot(x - center[0], y - center[1]) <= self.radius

    def on_mouse_down(self, pos, button):
        if self.collide_point(*pos):
            if self.disabled:
                return True
            if self.callback is not None:
                if self.callback(pos, button):
                    return True

    def _draw(self, win: pygame.Surface, dt: int):
        pygame.draw.circle(win, self.border_color, self.center, self.radius)
        pygame.draw.circle(win, self.color, self.center, self.radius - self.border_width)


class OthelloWorld(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.game = kwargs['game']  # type: Game

        self._setup_ui()

    def _setup_ui(self):
        # Create sprites.
        _piece_radius = (CONF.U.BOARD_EDGES['right'] - CONF.U.BOARD_EDGES['left']) / CONF.G.BOARD_WIDTH / 2 - 5.0

        def _piece_pos(i, j):
            return (CONF.U.BOARD_EDGES['left'] + (CONF.U.BOARD_EDGES['right'] - CONF.U.BOARD_EDGES['left']) /
                    CONF.G.BOARD_WIDTH * (j + 0.5),
                    CONF.U.BOARD_EDGES['top'] + (CONF.U.BOARD_EDGES['bottom'] - CONF.U.BOARD_EDGES['top']) /
                    CONF.G.BOARD_HEIGHT * (i + 0.5))

        def _callback_factory(i, j):
            def _callback(pos, button, i=i, j=j):
                self.game.take_action({
                    'type': 'play',
                    'i': i, 'j': j,
                    'player_id': self.game.state.current_player_id,
                })
                return True
            return _callback

        self.board_pieces = [[      # type: List[List[ChessPiece]]
            self.add_child(ChessPiece(
                center=_piece_pos(i, j),
                width=2 * _piece_radius, height=2 * _piece_radius,
                color=THECOLORS['white'], border_width=2.5,
                visible=False, callback=_callback_factory(i, j),
            ))
            for j in range(CONF.G.BOARD_WIDTH)
        ] for i in range(CONF.G.BOARD_HEIGHT)]

        self.players = {
            0: Label(
                text='玩家0', font_size=48,
                color=THECOLORS['red'],
                center=CONF.U.PLAYER_LABEL_POS[0],
            ),
            1: Label(
                text='玩家1', font_size=48,
                color=THECOLORS['blue'],
                center=CONF.U.PLAYER_LABEL_POS[1],
            ),
        }
        for player in self.players.values():
            self.add_child(player)

        self.single_start_btn = self.add_child(ReverseColorButton(
            text='单机启动', font_size=48,
            width=48 * 5 + 8, height=48 + 8,
            callback=self.game.single_start,
            center=CONF.U.PLAY_BTN_POS['single'],
        ))
        self.start_server_btn = self.add_child(ReverseColorButton(
            text='启动服务器', font_size=48,
            width=48 * 5 + 8, height=48 + 8,
            callback=self.game.start_server,
            center=CONF.U.PLAY_BTN_POS['server'],
        ))
        self.join_game_btn = self.add_child(ReverseColorButton(
            text='加入游戏', font_size=48,
            width=48 * 5 + 8, height=48 + 8,
            callback=self.game.join_game,
            center=CONF.U.PLAY_BTN_POS['join'],
        ))
        self.quit_game_btn = self.add_child(ReverseColorButton(
            text='退出游戏', font_size=48,
            width=48 * 5 + 8, height=48 + 8,
            callback=self.game.exit_game,
            center=CONF.U.PLAY_BTN_POS['exit'],
        ))

    def on_update(self, state_dict):
        state = self.game.state
        for i in range(len(state.board)):
            for j in range(len(state.board[0])):
                cell = state.board[i][j]
                piece = self.board_pieces[i][j]

                if cell is None:
                    piece.visible = False
                else:
                    piece.visible = True
                    piece.disabled = True
                    piece.color = state.players[cell].color

    def _draw(self, win: pygame.Surface, dt: int):
        win.fill(THECOLORS['white'])

        # Board
        for j in range(CONF.G.BOARD_WIDTH + 1):
            x_pos = CONF.U.BOARD_EDGES['left'] + j * (CONF.U.BOARD_EDGES['right'] - CONF.U.BOARD_EDGES['left']) / CONF.G.BOARD_WIDTH
            pygame.draw.line(
                win, CONF.U.BOARD_EDGE_COLOR,
                (x_pos, CONF.U.BOARD_EDGES['top']),
                (x_pos, CONF.U.BOARD_EDGES['bottom']),
                CONF.U.BOARD_LINE_WIDTH,
            )
        for i in range(CONF.G.BOARD_HEIGHT + 1):
            y_pos = CONF.U.BOARD_EDGES['top'] + i * (CONF.U.BOARD_EDGES['bottom'] - CONF.U.BOARD_EDGES['top']) / CONF.G.BOARD_HEIGHT
            pygame.draw.line(
                win, CONF.U.BOARD_EDGE_COLOR,
                (CONF.U.BOARD_EDGES['left'], y_pos),
                (CONF.U.BOARD_EDGES['right'], y_pos),
                CONF.U.BOARD_LINE_WIDTH,
            )

        for child in self.children[:]:
            draw_fn = getattr(child, 'draw', None)
            if draw_fn is not None:
                draw_fn(win, dt)


class UI:
    def __init__(self, game: 'Game', width, height, caption='Game'):
        self.game = game
        self.width = width
        self.height = height
        self.caption = caption
        self.window: pygame.Surface
        self.clock: pygame.time.Clock
        self.root: OthelloWorld

    def initialize(self):
        pygame.init()
        self.window = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(self.caption)

        self.root = OthelloWorld(game=self.game, width=self.width, height=self.height)

        self.clock = pygame.time.Clock()

    def finalize(self):
        pygame.quit()

    def dispatch_pygame_event(self, event: pygame.event.Event):
        return self.root.dispatch_pygame_event(event)

    def draw(self, dt):
        self.root.draw(self.window, dt)
        pygame.display.update()


class Game:
    """Game = UI(EventDispatcher) + State + Client + Server (optional)"""
    def __init__(self):
        self.ui = UI(self, CONF.U.WIDTH, CONF.U.HEIGHT, 'Othello')
        self.state = logic.GameState()
        self.server = None
        self.client = None
        self.status = 'idle'

    def initialize(self):
        self.ui.initialize()
        self.status = 'waiting'

    def finalize(self):
        self.status = 'idle'
        self.ui.finalize()

    def run(self):
        self.initialize()

        try:
            while self.status in {'idle', 'waiting', 'running'}:
                dt = self.ui.clock.tick(CONF.U.FPS)

                for event in pygame.event.get():  # type: pygame.event.Event
                    if event.type == pygame.QUIT:
                        if self.ui.dispatch_pygame_event(event):
                            # SIGTERM processed by widgets
                            continue
                        self.status = 'stopped'
                    else:
                        self.ui.dispatch_pygame_event(event)

                self.ui.draw(dt)
        finally:
            self.finalize()

    def take_action(self, action: dict):
        if action['type'] != 'init' and self.status != 'running':
            return

        server_reply = self.client.send_action(action)
        self.state.load_state_dict(server_reply)
        utils.push_event(CONF.G.UPDATE_EVENT, state_dict=server_reply)

    def single_start(self, pos, button):
        self.server = server.LocalServer()
        self.client = client.LocalClient(self.server)
        self.take_action({'type': 'init', 'ip_address': '', 'name': '玩家0'})
        self.take_action({'type': 'init', 'ip_address': '', 'name': '玩家1'})
        self.status = 'running'

    def start_server(self, pos, button):
        print('Starting server')

    def join_game(self, pos, button):
        print('Join game')

    def exit_game(self, pos, button):
        utils.push_event(pygame.QUIT)


def main():
    game = Game()
    game.run()


if __name__ == '__main__':
    main()
