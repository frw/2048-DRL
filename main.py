# -*- coding: UTF-8 -*-
from __future__ import print_function

import argparse
import sys

from tui.game import Game
from rl.reinforcement_learner import StandardQLearner
from rl.reinforcement_learner import DeepQLearner


def print_rules_and_exit():
    print("""Use your arrow keys to move the tiles.
When two tiles with the same value touch they merge into one with the sum of
their value! Try to reach 2048 to win.""")
    sys.exit(0)


def parse_cli_args():
    """parse args from the CLI and return a dict"""
    parser = argparse.ArgumentParser(description='2048 in your terminal')
    parser.add_argument('--mode', dest='mode', type=str,
                        default=None, help='Color mode (dark or light)')
    parser.add_argument('--az', dest='azmode', action='store_true',
                        help='Use the letters a-z instead of numbers')
    parser.add_argument('--ai', dest='ai', action='store_true',
                        help='Play the game using an AI')
    parser.add_argument('-r', '--rules', action='store_true')
    return vars(parser.parse_args())


def start_game():
    """
    Start a new game. If ``debug`` is set to ``True``, the game object is
    returned and the game loop isn't fired.
    """

    ###this section temporarily changed by robert
    all_scores = [] 
    weight_tracker = [] 
    other_weight_tracker = [] 
    another_weight_tracker = []
    ######

    args = parse_cli_args()

    if args['rules']:
        print_rules_and_exit()

    if args['ai']:
        ai = DeepQLearner().load()
        args['ai'] = ai
    else:
        ai = None
        del args['ai']

    if ai is not None:
        saved = False
        while True:
            ai.new_epoch()

            game = Game(**args)
            score = game.loop()
            if score is None:
                break

            ai.end_epoch(score)
            saved = False

            ###this section temporarily changed by robert
            all_scores.append(score) #robert
            weight_tracker.append(float(ai.network.get_all_weights()[2][7]))
            other_weight_tracker.append(ai.network.get_all_weights()[0][3,8])
            another_weight_tracker.append(ai.network.get_all_weights()[1][2])
            if ai.epoch % 200 == 0: #changed by robert
                ai.save()
                saved = True
                print (all_scores) #robert
                print (weight_tracker)
                print (other_weight_tracker)
                print (another_weight_tracker)
                break #added by robert
            #####

        if not saved:
            ai.save()
    else:
        game = Game(**args)
        game.loop()

if __name__ == "__main__":
    start_game()
