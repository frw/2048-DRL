'''
Game class, for all aspects related to an instance of playing the 2048 Game.
'''

from __future__ import print_function

import atexit
import math
import os
import os.path
import sys

from colorama import init, Fore, Style

import keypress
from board import Board
init(autoreset=True)


class Game(object):
    """
    A 2048 game
    """
    __dirs = {
        keypress.UP:      Board.UP,
        keypress.DOWN:    Board.DOWN,
        keypress.LEFT:    Board.LEFT,
        keypress.RIGHT:   Board.RIGHT,
    }

    __is_windows = os.name == 'nt'

    COLORS = {
        2:      Fore.RED,
        4:      Fore.GREEN,
        8:      Fore.YELLOW,
        16:     Fore.BLUE + Style.BRIGHT,
        32:     Fore.MAGENTA,
        64:     Fore.CYAN,
        128:    Fore.RED,
        256:    Fore.GREEN,
        512:    Fore.YELLOW,
        1024:   Fore.BLUE + Style.BRIGHT,
        2048:   Fore.MAGENTA,
        # just in case people set an higher goal they still have colors
        4096:   Fore.CYAN,
        8192:   Fore.RED,
        16384:  Fore.GREEN,
        32768:  Fore.YELLOW,
        65536:  Fore.BLUE + Style.BRIGHT,
        131072: Fore.MAGENTA,
    }

    # these are color replacements for various modes
    __color_modes = {
        'dark': {
            Fore.BLUE: Fore.WHITE,
            Fore.BLUE + Style.BRIGHT: Fore.WHITE,
        },
        'light': {
            Fore.YELLOW: Fore.BLACK,
        },
    }

    SCORES_FILE = '.highscore'

    def __init__(self, scores_file=SCORES_FILE, colors=COLORS,
                 clear_screen=True, mode=None, azmode=False,
                 ai=None, **kws):
        """
        Create a new game.
            scores_file: file to use for the best score (default
                         is ~/.term2048.scores)
            colors: dictionary with colors to use for each tile
            mode: color mode. This adjust a few colors and can be 'dark' or
                  'light'. See the adjustColors functions for more info.
            other options are passed to the underlying Board object.
        """
        self.board = Board(**kws)
        self.score = 0
        self.best_score = 0
        self.__scores_file = scores_file
        self.__clear_screen = clear_screen

        self.__colors = colors
        self.__azmode = azmode
        self.__az = {}
        for i in range(1, int(math.log(self.board.goal(), 2))):
            self.__az[2 ** i] = chr(i + 96)

        self.__ai = ai

        self.load_best_score()
        self.adjust_colors(mode)

    def adjust_colors(self, mode='dark'):
        """
        Change a few colors depending on the mode to use. The default mode
        doesn't assume anything and avoid using white & black colors. The dark
        mode use white and avoid dark blue while the light mode use black and
        avoid yellow, to give a few examples.
        """
        rp = Game.__color_modes.get(mode, {})
        for k, color in self.__colors.items():
            self.__colors[k] = rp.get(color, color)

    def load_best_score(self):
        """
        load local best score from the default file
        """
        try:
            with open(self.__scores_file, 'r') as f:
                self.best_score = int(f.readline(), 10)
        except IOError:
            self.best_score = 0
            return False
        return True

    def save_best_score(self):
        """
        save current best score in the default file
        """
        if self.score > self.best_score:
            self.best_score = self.score
        try:
            with open(self.__scores_file, 'w') as f:
                f.write(str(self.best_score))
        except IOError:
            return False
        return True

    def increment_score(self, pts):
        """
        update the current score by adding it the specified number of points
        :param pts: amount to increase score by
        """
        self.score += pts
        if self.score > self.best_score:
            self.best_score = self.score

    @staticmethod
    def read_move():
        """
        read and return a move to pass to a board
        """
        k = keypress.get_key()
        return Game.__dirs.get(k)

    def clear_screen(self):
        """Clear the console"""
        if self.__clear_screen:
            os.system('cls' if self.__is_windows else 'clear')
        else:
            print('\n')

    def hide_cursor(self):
        """
        Hide the cursor. Don't forget to call ``showCursor`` to restore
        the normal shell behavior. This is a no-op if ``clear_screen`` is
        falsy.
        """
        if not self.__clear_screen:
            return
        if not self.__is_windows:
            sys.stdout.write('\033[?25l')

    def show_cursor(self):
        """Show the cursor."""
        if not self.__is_windows:
            sys.stdout.write('\033[?25h')

    def loop(self):
        """
        main game loop. returns the final score.
        """
        margins = {'left': 4, 'top': 4, 'bottom': 4}
        move_str = {
            Board.UP: 'Up',
            Board.DOWN: 'Down',
            Board.LEFT: 'Left',
            Board.RIGHT: 'Right'
        }

        atexit.register(self.show_cursor)

        try:
            self.hide_cursor()
            change = None
            can_move = True
            while True:
                self.clear_screen()
                print(self.__str__(margins=margins, change=change))
                if not can_move:
                    if self.__ai is not None:
                        self.__ai.action_callback(self.board.cells, None)
                    break

                if self.__ai is not None:
                    m = self.__ai.action_callback(self.board.cells,
                                                  self.board.possible_moves())
                else:
                    m = None
                    while m is None:
                        m = self.read_move()

                score_inc = self.board.move(m)
                self.increment_score(score_inc)
                change = (score_inc, move_str.get(m))

                can_move = self.board.can_move()
                if not can_move:
                    score_inc -= 10000

                if self.__ai is not None:
                    self.__ai.reward_callback(score_inc)

        except KeyboardInterrupt:
            self.save_best_score()
            return None

        self.save_best_score()
        print('You won!' if self.board.won() else 'Game Over')

        print (self.score)

        return self.score

    def get_cell_str(self, x, y):
        """
        return a string representation of the cell located at x,y.
        """
        c = self.board.get_cell(x, y)

        if c == 0:
            return '.' if self.__azmode else '  .'

        elif self.__azmode:
            if c not in self.__az:
                s = '?'
            else:
                s = self.__az[c]
        elif c >= 1024:
            s = '%2dk' % (c / 1024)
        else:
            s = '%3d' % c

        return self.__colors.get(c, Fore.RESET) + s + Style.RESET_ALL

    def board_to_str(self, margins={}):
        """
        return a string representation of the current board.
        """
        b = self.board
        rg = range(b.size())
        left = ' ' * margins.get('left', 0)
        s = '\n'.join(
            [left + ' '.join([self.get_cell_str(x, y) for x in rg]) for y in rg])
        return s

    def __str__(self, margins={}, change=None):
        top = '\n' * margins.get('top', 0)
        bottom = '\n' * margins.get('bottom', 0)
        left = ' ' * margins.get('left', 0)
        board = self.board_to_str(margins=margins)
        scores = left + 'Score: %7d  Best: %7d\n' % (self.score, self.best_score)
        changes = '\n' if change is None else left + '+%13d  %s\n' % change
        output = top + board + '\n\n' + scores + changes + bottom

        if self.__ai is None:
            return output
        else:
            return ('Epoch: %d\n' % self.__ai.epoch) + output
