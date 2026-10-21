"""Kinthread reasoning: the fact language the reader model writes, and the exact engine that answers from it."""
from .engine import World, relation_in, solve
from .facts import format, parse
from .kin import KIN, name_moves
