#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import antlr4

from generated.Fly.FlyLexer import FlyLexer
from generated.Fly.FlyParser import FlyParser

SAMPLE_PATH = Path(__file__).absolute().parent / 'samples' / 'fly-sample.txt'


def main():
    input_stream = antlr4.FileStream(str(SAMPLE_PATH))
    lexer = FlyLexer(input_stream)
    stream = antlr4.CommonTokenStream(lexer)
    parser = FlyParser(stream)
    tree = parser.program()
    print(tree.toStringTree())


if __name__ == '__main__':
    main()
