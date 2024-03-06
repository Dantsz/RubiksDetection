
from enum import Enum


class SquareColor(str, Enum):
    """Enum to represent the color of a square in a face of the cube."""
    Unknown = 0
    WHITE = 1
    YELLOW = 2
    BLUE = 3
    GREEN = 4
    RED = 5
    ORANGE = 6
