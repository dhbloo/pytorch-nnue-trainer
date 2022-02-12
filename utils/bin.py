import ctypes
from enum import Enum


class Result(Enum):
    LOSS = 0
    DRAW = 1
    WIN = 2

    def opposite(r):
        return Result(2 - r.value)


class Rule(Enum):
    FREESTYLE = 0
    STANDARD = 1
    RENJU = 4


class Move():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def value(self):
        return (self.x << 5) | self.y

    def __repr__(self):
        return f'({self.x},{self.y})'

    def to_pos(self):
        return chr(self.x + ord("a")) + str(self.y + 1)


class Entry(ctypes.Structure):
    _fields_ = [('result', ctypes.c_uint16, 2), ('ply', ctypes.c_uint16, 9),
                ('boardsize', ctypes.c_uint16, 5), ('rule', ctypes.c_uint16, 3),
                ('move', ctypes.c_uint16, 13), ('position', ctypes.c_uint16 * 1024)]


class EntryHead(ctypes.Structure):
    _fields_ = [('result', ctypes.c_uint16, 2), ('ply', ctypes.c_uint16, 9),
                ('boardsize', ctypes.c_uint16, 5), ('rule', ctypes.c_uint16, 3),
                ('move', ctypes.c_uint16, 13)]


def write_entry(f, result: Result, boardsize: int, rule: Rule, move: Move, position: list):
    entry = Entry()
    entry.result = result.value
    entry.ply = len(position)
    entry.boardsize = boardsize
    entry.rule = rule.value
    entry.move = move.value()
    for i, m in enumerate(position):
        entry.position[i] = m.value()

    f.write(bytearray(entry)[:4 + 2 * len(position)])


def read_entry(f):
    ehead = EntryHead()
    f.readinto(ehead)

    result = Result(ehead.result)
    ply = int(ehead.ply)
    boardsize = int(ehead.boardsize)
    rule = Rule(ehead.rule)
    move = Move((ehead.move >> 5) & 31, ehead.move & 31)

    pos_array = (ctypes.c_uint16 * ehead.ply)()
    f.readinto(pos_array)
    position = [Move((m >> 5) & 31, m & 31) for m in pos_array]

    return result, ply, boardsize, rule, move, position

