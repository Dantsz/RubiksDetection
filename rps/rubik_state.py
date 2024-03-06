from enum import Enum

from rpd.metafeatures import Face

class SquareColor(str, Enum):
    """Enum to represent the color of a square in a face of the cube."""
    Unknown = 0
    WHITE = 1
    YELLOW = 2
    BLUE = 3
    GREEN = 4
    RED = 5
    ORANGE = 6

class RubikStateEngine:
    """Represents the state of a Rubik's cube.

    It can be fed with the faces from the detection engine and it will keep track of the state of the cube.
    """

    def __init__(self):
        pass

    def consume_face(self, face: Face):
        """Consumes a face from the detection engine and updates the state of the cube."""
        pass
