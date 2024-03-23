
import numpy as np
from .color import SquareColor

class CubeState:
    """Represents the state of a Rubik's cube.

    The cube state is represented as a 6x9 matrix, where each row represents a face of the cube.
    The squares are represented as integers, where 0 is the top-left square and 8 is the bottom-right square.
    """
    state: np.ndarray

    def __init__(self, state: np.ndarray):
        assert state.shape == (6, 3, 3), "The state must be a 6x3x3"
        self.state = state

    def to_solver_string(self) -> str:
        """Returns the state of the cube as a string that can be used as input for the Kociemba solver.

        The order of the center colors is as follows:
        W , G , R , B , O , Y
        """
        string = ""
        for face in self.state:
            for row in face:
                for square in row:
                    string += SquareColor(square).name[0]
        return string