
import numpy as np
from .color import SquareColor

class CubeState:
    """Represents the state of a Rubik's cube.

    The cube state is represented as a 6x9 matrix, where each row represents a face of the cube.
    The squares are represented as integers, where 0 is the top-left square and 8 is the bottom-right square.
    """
    state: np.ndarray

    def to_solver_string(self) -> str:
        """Returns the state of the cube as a string that can be used as input for the Kociemba solver.

        The order of the center colors is as follows:
        W , G , R , B , O , Y
        """
        return ""