# -*- coding: UTF-8 -*-
from __future__ import print_function

import argparse
import sys

from tui.game import Game


def print_rules_and_exit():
    print("""Use your arrow keys to move the tiles.
When two tiles with the same value touch they merge into one with the sum of
their value! Try to reach 2048 to win.""")
    sys.exit(0)


def parse_cli_args():
    """parse args from the CLI and return a dict"""
    parser = argparse.ArgumentParser(description='2048 in your terminal')
    parser.add_argument('--mode', dest='mode', type=str,
                        default=None, help='colors mode (dark or light)')
    parser.add_argument('--az', dest='azmode', action='store_true',
                        help='Use the letters a-z instead of numbers')
    parser.add_argument('--resume', dest='resume', action='store_true',
                        help='restart the game from where you left')
    parser.add_argument('-r', '--rules', action='store_true')
    return vars(parser.parse_args())


def start_game():
    """
    Start a new game. If ``debug`` is set to ``True``, the game object is
    returned and the game loop isn't fired.
    """
    args = parse_cli_args()

    if args['rules']:
        print_rules_and_exit()

    game = Game(**args)
    if args['resume']:
        game.restore()

    return game.loop()

if __name__ == "__main__":
    start_game()