#! /usr/bin/python
# -*- coding: utf-8 -*-

import pygame


def push_event(event_type, *event_args, **event_kwargs):
    if event_args is None:
        event_args = {}
    pygame.event.post(pygame.event.Event(event_type, *event_args, **event_kwargs))
