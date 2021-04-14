#! /usr/bin/python
# -*- coding: utf-8 -*-

import math
from copy import deepcopy
from typing import List, Dict, Tuple

import pygame

from constants import *


class EventDispatcher:
    def __init__(self):
        self.children = []

    def add_child(self, child: 'EventDispatcher'):
        self.children.append(child)
        return child

    def dispatch(self, event_name, *args, **kwargs):
        method = getattr(self, event_name, None)
        if method is not None:
            return method(*args, **kwargs)

    def dispatch_pygame_event(self, event: pygame.event.Event):
        print(event)
        pass


class Widget(EventDispatcher):
    def __init__(self, **kwargs):
        super().__init__()
        self.disabled = False
        self.visible = kwargs.get('visible', True)
        self.x = kwargs.get('x', 0)
        self.y = kwargs.get('y', 0)
        self.width = kwargs.get('width', 100)
        self.height = kwargs.get('height', 100)

    @property
    def pos(self):
        return self.x, self.y

    @property
    def right(self):
        return self.x + self.width

    @property
    def bottom(self):
        return self.y + self.height

    @property
    def center(self):
        return self.x + self.width / 2, self.y + self.height / 2

    @center.setter
    def center(self, pos):
        x, y = pos
        self.x = x - self.width / 2
        self.y = y - self.height / 2

    def collide_point(self, x, y):
        return self.x <= x <= self.right and self.y <= y <= self.bottom

    def on_mouse_down(self, pos, button):
        if self.disabled and self.collide_point(*pos):
            return True
        for child in self.children[:]:
            if child.dispatch('on_mouse_down', pos, button):
                return True

    def on_mouse_motion(self, pos, rel, buttons):
        if self.disabled and self.collide_point(*pos):
            return True
        for child in self.children[:]:
            if child.dispatch('on_mouse_motion', pos, rel, buttons):
                return True

    def on_mouse_up(self, pos, button):
        if self.disabled and self.collide_point(*pos):
            return True
        for child in self.children[:]:
            if child.dispatch('on_mouse_up', pos, button):
                return True

    def draw(self, win: pygame.Surface, dt: int):
        if not self.visible:
            return
        self._draw(win, dt)

    def _draw(self, win: pygame.Surface, dt: int):
        pass


class Label(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.text = kwargs.get('text', '')
        self.color = kwargs.get('color', THECOLORS['black'])
        self.bgcolor = kwargs.get('bgcolor', THECOLORS['white'])

        text_surface = self._render_text()
        self.width = min(text_surface.get_width(), self.width)
        self.height = min(text_surface.get_height(), self.height)

    def _render_text(self) -> pygame.Surface:
        return CONF.U.FONT.render(self.text, True, self.color, self.bgcolor)

    def _draw(self, win: pygame.Surface, dt: int):
        text = self._render_text()
        win.blit(text, self.pos)

    def on_mouse_down(self, pos, button):
        if self.collide_point(*pos):
            print(f'| {self.text} clicked!')


class ChessPiece(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.center = kwargs.get('center', self.center)
        self.color = kwargs.get('color', THECOLORS['black'])
        self.border_width = kwargs.get('border_width', 2)
        self.border_color = kwargs.get('border_color', THECOLORS['black'])

        if self.width != self.height:
            raise ValueError('width != height for chess piece')

    @property
    def radius(self):
        return self.width / 2

    def collide_point(self, x, y):
        center = self.center
        return math.hypot(x - center[0], y - center[1]) <= self.radius

    def _draw(self, win: pygame.Surface, dt: int):
        pygame.draw.circle(win, self.border_color, self.center, self.radius)
        pygame.draw.circle(win, self.color, self.center, self.radius - self.border_width)


class OthelloWorld(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Create sprites.
        _piece_radius = (CONF.U.BOARD_EDGES['right'] - CONF.U.BOARD_EDGES['left']) / CONF.G.BOARD_WIDTH / 2 - 5.0

        def _piece_pos(i, j):
            return (CONF.U.BOARD_EDGES['left'] + (CONF.U.BOARD_EDGES['right'] - CONF.U.BOARD_EDGES['left']) / CONF.G.BOARD_WIDTH * (i + 0.5),
                    CONF.U.BOARD_EDGES['top'] + (CONF.U.BOARD_EDGES['bottom'] - CONF.U.BOARD_EDGES['top']) / CONF.G.BOARD_HEIGHT * (j + 0.5))

        self.board_pieces = [[      # type: List[List[ChessPiece]]
            self.add_child(ChessPiece(
                center=_piece_pos(i, j),
                width=2 * _piece_radius, height=2 * _piece_radius,
                color=THECOLORS['white'], border_width=2.5,
                visible=False,
            ))
            for i in range(CONF.G.BOARD_WIDTH)
        ] for j in range(CONF.G.BOARD_HEIGHT)]

        self.players = {
            0: Label(
                text='Player 0',
                color=THECOLORS['red'],
            ),
            1: Label(
                text='Player 1',
                color=THECOLORS['blue'],
            ),
        }
        self.players[0].center = CONF.U.PLAYER_LABEL_POS[0]
        self.players[1].center = CONF.U.PLAYER_LABEL_POS[1]
        for player in self.players.values():
            self.add_child(player)

    def update(self, game_state: 'GameState'):
        for i in range(len(game_state.board)):
            for j in range(len(game_state.board[0])):
                cell = game_state.board[i][j]
                piece = self.board_pieces[i][j]

                if cell is None:
                    piece.visible = False
                    piece.disabled = True
                else:
                    piece.visible = True
                    piece.disabled = False
                    piece.color = game_state.players[cell].color

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
    def __init__(self, width, height, caption='Game'):
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

        CONF.U.FONT = pygame.font.Font(CONF.U.FONT_NAME, CONF.U.FONT_SIZE)

        self.root = OthelloWorld(width=self.width, height=self.height)

        self.clock = pygame.time.Clock()

    def finalize(self):
        pygame.display.quit()
        pygame.quit()

    def dispatch_pygame_event(self, event: pygame.event.Event):
        return self.root.dispatch_pygame_event(event)

    def update(self, game_state: 'GameState'):
        self.root.update(game_state)

    def draw(self, dt):
        self.root.draw(self.window, dt)
        pygame.display.update()


class PlayerInfo:
    def __init__(self, player_id: int, name: str, color: Tuple, status: str):
        self.player_id = player_id
        self.name = name
        self.color = color
        self.status = status

    def state_dict(self):
        return {
            'player_id': self.player_id,
            'name': self.name,
            'color': self.color,
            'status': self.status,
        }


class GameState:
    def __init__(self):
        self.board = [[None for _ in range(CONF.G.BOARD_WIDTH)] for _ in range(CONF.G.BOARD_HEIGHT)]
        self.current_player = 0
        self.players = {}   # type: Dict[int, PlayerInfo]

    def initialize(self):
        pass

    def finalize(self):
        pass

    def state_dict(self):
        return {
            'current_player': self.current_player,
            'board': self.board,
            'players': {player_id: info.state_dict() for player_id, info in self.players.items()},
        }

    def load_state_dict(self, state_dict: dict):
        self.current_player = state_dict['current_player']
        self.board = deepcopy(state_dict['board'])
        self.players = {player_id: PlayerInfo(**info_state_dict)
                        for player_id, info_state_dict in state_dict['players'].items()}


class ServerGameState(GameState):
    def __init__(self):
        super().__init__()

    def new_player(self):
        pass


class Game:
    """Game = UI(EventDispatcher) + State + Connection"""
    def __init__(self):
        self.ui = UI(CONF.U.WIDTH, CONF.U.HEIGHT, 'Othello')
        self.state = GameState()
        self.connection = None
        self.status = 'idle'

    def initialize(self):
        self.state.initialize()
        self.ui.initialize()
        self.status = 'running'

    def finalize(self):
        self.status = 'idle'
        self.ui.finalize()
        self.state.finalize()

    def run(self):
        self.initialize()

        try:
            while self.status == 'running':
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


def main():
    game = Game()
    game.run()


if __name__ == '__main__':
    main()
