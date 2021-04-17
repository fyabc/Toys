#! /usr/bin/python
# -*- coding: utf-8 -*-

"""A simple framework of pygame event dispatching."""

import pygame
from pygame.color import THECOLORS

from LearnMultiplayerGame import draw_utils as du
from LearnMultiplayerGame.constants import CONF


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
    def center_x(self):
        return self.x + self.width / 2

    @center_x.setter
    def center_x(self, value):
        self.x = value - self.width / 2

    @property
    def center_y(self):
        return self.y + self.height / 2

    @center_y.setter
    def center_y(self, value):
        self.y = value - self.height / 2

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
