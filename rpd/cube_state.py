
from typing import Self
import numpy as np
from .color import SquareColor, color_to_spatial_symbol

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
            for col in range(0, 3):
                for row in range(0, 3):
                    string += color_to_spatial_symbol(SquareColor(face[row][col]))
        return string

    def is_solved(self) -> bool:
        """Returns True if the cube is solved."""
        # for face in self.state:
        #     if
        pass

    def rotate_clockwise(self, face: SquareColor) -> Self:
        """Rotates the face of the cube clockwise."""
        raise NotImplementedError("This method is not implemented yet.")
        return self
    def rotate_counter_clockwise(self, face: SquareColor) -> Self:
        """Rotates the face of the cube counterclockwise."""
        raise NotImplementedError("This method is not implemented yet.")
        return self
    def rotate_twice(self, face: SquareColor) -> Self:
        """Rotates the face of the cube twice."""
        return self.rotate_clockwise(face).rotate_clockwise(face)
