# -*- coding: UTF-8 -*-
import random

# PY3 compat
try:
    xrange
except NameError:
    xrange = range


class Board(object):
    """
    A 2048 board
    """

    UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3

    GOAL = 2048
    SIZE = 4

    def __init__(self, goal=GOAL, size=SIZE, **kws):
        self.__size = size
        self.__size_range = xrange(0, self.__size)
        self.__goal = goal
        self.__won = False
        self.cells = [[0] * self.__size for _ in xrange(self.__size)]
        self.add_tile()
        self.add_tile()

    def size(self):
        """return the board size"""
        return self.__size

    def goal(self):
        """return the board goal"""
        return self.__goal

    def won(self):
        """
        return True if the board contains at least one tile with the board goal
        """
        return self.__won

    def can_move(self):
        """
        test if a move is possible
        """
        for y in self.__size_range:
            for x in self.__size_range:
                c = self.get_cell(x, y)
                if c == 0:
                    return True
                if (x < self.__size - 1 and c == self.get_cell(x + 1, y)) \
                        or (y < self.__size - 1 and c == self.get_cell(x, y + 1)):
                    return True

        return False

    def add_tile(self, value=None, choices=([2] * 9 + [4])):
        """
        add a random tile in an empty cell
        :param value: value of the tile to add.
        :param choices: a list of possible choices for the value of the tile.
                   default is [2, 2, 2, 2, 2, 2, 2, 2, 2, 4].
        """
        if value:
            choices = [value]

        v = random.choice(choices)
        empty = [(x, y)
                 for x in self.__size_range
                 for y in self.__size_range
                 if self.get_cell(x, y) == 0]
        if empty:
            x, y = random.choice(empty)
            self.set_cell(x, y, v)

    def get_cell(self, x, y):
        """return the cell value at x,y"""
        return self.cells[y][x]

    def set_cell(self, x, y, v):
        """set the cell value at x,y"""
        self.cells[y][x] = v

    def get_line(self, y):
        """return the y-th line, starting at 0"""
        return self.cells[y]

    def get_col(self, x):
        """return the x-th column, starting at 0"""
        return [self.get_cell(x, i) for i in self.__size_range]

    def set_line(self, y, l):
        """set the y-th line, starting at 0"""
        self.cells[y] = l[:]

    def set_col(self, x, l):
        """set the x-th column, starting at 0"""
        for i in xrange(0, self.__size):
            self.set_cell(x, i, l[i])

    def possible_moves(self):
        """return a list of possible moves for the current board"""
        moves = set()

        for y in self.__size_range:
            for x in self.__size_range:
                c = self.get_cell(x, y)

                if c == 0:
                    continue

                if y > 0:
                    up = self.get_cell(x, y - 1)
                    if up == c or up == 0:
                        moves.add(Board.UP)

                if y < self.__size - 1:
                    down = self.get_cell(x, y + 1)
                    if down == c or down == 0:
                        moves.add(Board.DOWN)

                if x > 0:
                    left = self.get_cell(x - 1, y)
                    if left == c or left == 0:
                        moves.add(Board.LEFT)

                if x < self.__size - 1:
                    right = self.get_cell(x + 1, y)
                    if right == c or right == 0:
                        moves.add(Board.RIGHT)

        return list(moves)

    def __move_and_merge(self, line, d):
        """
        Merge tiles in a line or column according to a direction and return a
        tuple with the new line, the score for the move on this line, and
        whether the move has changed the line
        """
        if d == Board.LEFT or d == Board.UP:
            next_insert = 0
            inc = 1
            rg = xrange(next_insert, self.__size, inc)
        else:
            next_insert = self.__size - 1
            inc = -1
            rg = xrange(next_insert, -1, inc)

        pts = 0
        changed = False
        previous = None
        for i in rg:
            current = line[i]

            if current == 0:
                continue
            elif previous is None or line[previous] != current:
                if next_insert != i:
                    line[next_insert] = current
                    line[i] = 0
                    changed = True
                previous = next_insert
                next_insert += inc
            else:
                v = current * 2

                if v == self.__goal:
                    self.__won = True

                pts += v
                changed = True
                line[previous] = current * 2
                line[i] = 0
                previous = None

        return line, pts, changed

    def move(self, d, add_tile=True):
        """
        move and return the move score
        """
        if d == Board.LEFT or d == Board.RIGHT:
            chg, get = self.set_line, self.get_line
        elif d == Board.UP or d == Board.DOWN:
            chg, get = self.set_col, self.get_col
        else:
            return 0

        moved = False
        score = 0

        for i in self.__size_range:
            # save the original line/col
            original = get(i)
            # move and merge adjacent tiles
            new, pts, changed = self.__move_and_merge(original, d)
            if changed:
                # set it back in the board
                chg(i, new)
                moved = True
            score += pts

        # don't add a new tile if nothing changed
        if moved and add_tile:
            self.add_tile()

        return score
