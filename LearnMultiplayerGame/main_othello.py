#! /usr/bin/python
# -*- coding: utf-8 -*-

from ui import World
from logic import GameState
from draw_utils import main_loop


class Game:
    def __init__(self):
        self.state = GameState()
        self.world = World(self.state)
        # self.client = Client()

    def update(self, dt):
        self.world.update(dt)


def main():
    game = Game()
    main_loop(game)


if __name__ == '__main__':
    main()
