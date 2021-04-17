#! /usr/bin/python
# -*- coding: utf-8 -*-

import math
from typing import List, Dict, Tuple

import pygame

import draw_utils as du
import logic
import utils
import server
import client
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
        # Kill application
        if event.type == pygame.QUIT:
            self.dispatch('on_exit')

        # Mouse motion

        # Mouse action
        elif event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP):
            if event.type == pygame.MOUSEBUTTONDOWN:
                event_name = 'on_mouse_down'
            else:
                event_name = 'on_mouse_up'
            self.dispatch(event_name, event.pos, event.button)

        # User event
        elif event.type == CONF.G.UPDATE_EVENT:
            self.dispatch('on_update', event.state_dict)

        # unhandled event here
        # print(f'Unhandled event: {event}')


class Widget(EventDispatcher):
    def __init__(self, **kwargs):
        super().__init__()
        self.disabled = False
        self.visible = kwargs.get('visible', True)
        self.x = kwargs.get('x', 0)
        self.y = kwargs.get('y', 0)
        self.width = kwargs.get('width', 1)
        self.height = kwargs.get('height', 1)

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

    @property
    def rect(self):
        return pygame.rect.Rect(self.x, self.y, self.width, self.height)

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
    """
    NOTE: The 'width' and 'height' parameters passed to the label only set the minimum value.
    These values may be changed by the font.
    It is recommended to set the center.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.text = kwargs.get('text', '')
        self.font_name = kwargs.get('font_name', 'SimHei')
        self.font_size = kwargs.get('font_size', 48)
        self.color = kwargs.get('color', THECOLORS['black'])
        self.bgcolor = kwargs.get('bgcolor', THECOLORS['white'])

        self._render_text()     # Modify width and height

        center = kwargs.get('center', None)
        if center is not None:
            self.center = center

    def _render_text(self) -> pygame.Surface:
        font = du.get_font(self.font_name, self.font_size)
        text_surface = font.render(self.text, True, self.color, self.bgcolor)
        self.width = max(text_surface.get_width(), self.width)
        self.height = max(text_surface.get_height(), self.height)
        return text_surface

    def _draw(self, win: pygame.Surface, dt: int):
        text = self._render_text()
        blit_pos = self.center[0] - text.get_width() / 2, self.center[1] - text.get_height() / 2
        win.blit(text, blit_pos)


class Button(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.border_color = kwargs.get('border_color', THECOLORS['black'])
        self.border_width: int = kwargs.get('border_width', 2)
        self.border_radius: int = kwargs.get('border_radius', 1)
        self.callback = kwargs.get('callback', None)
        self._clicked = False

    @property
    def clicked(self):
        return self._clicked

    @clicked.setter
    def clicked(self, value):
        self._clicked = value

    def on_mouse_down(self, pos, button):
        if self.collide_point(*pos):
            self.clicked = True

    def on_mouse_up(self, pos, button):
        """NOTE: When mouse up, release all buttons."""
        if self.clicked and self.callback is not None:
            self.callback(pos, button)
        self.clicked = False

    def _draw(self, win: pygame.Surface, dt: int):
        super()._draw(win, dt)

        pygame.draw.rect(win, self.border_color, self.rect, self.border_width, self.border_radius)


class ReverseColorButton(Button):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._backup_colors = self.color, self.bgcolor
        self._backup_rev_colors = [du.reverse_color(c) for c in self._backup_colors]

    @Button.clicked.setter
    def clicked(self, value):
        self._clicked = value

        if self._clicked:
            self.color, self.bgcolor = self._backup_rev_colors
        else:
            self.color, self.bgcolor = self._backup_colors


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

        self.player_labels = {
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
        for player in self.player_labels.values():
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
        board = state.board
        for i, j, p in board.iter_board():
            piece = self.board_pieces[i][j]

            if p is None:
                piece.visible = False
            else:
                piece.visible = True
                piece.disabled = True
                piece.color = state.players[p].color

        counter = board.count_players()
        for i, label in self.player_labels.items():
            if self.game.status == 'running':
                label.text = f'玩家{i}：{counter[i]}'
            else:
                label.text = f'玩家{i}'

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
