
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
        self.state = state.copy()

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

    def get_face(self, face: SquareColor) -> np.ndarray:
        """Returns the face of the cube."""
        return self.state[int(face)]

    def is_solved(self) -> bool:
        """Returns True if the cube is solved."""
        for face_id in range(0, 6):
            if not np.all(self.state[face_id] == face_id):
                return False
        return True

    def get_face_line(self, face: int, col: int | None, row: int | None, reverted: bool = False) -> np.ndarray:
        """Returns a line of the face of the cube.
        """
        if row is not None:
            if reverted:
                return self.state[face][row][::-1]
            else:
                return self.state[face][row]
        elif col is not None:
            if reverted:
                return self.state[face][:, col][::-1]
            else:
                return self.state[face][:, col]
        else:
            raise ValueError("Either row or col must be provided.")

    def __set_face_line(self, face: int, col: int | None, row: int | None, line: np.ndarray) -> Self:
        """Sets a line of the face of the cube."""
        if row is not None:
            self.state[face][row] = line
        elif col is not None:
            self.state[face][:, col] = line
        else:
            raise ValueError("Either row or col must be provided.")
        return self

    def rotate_clockwise_once(self, face: SquareColor) -> Self:
        """Rotates the face of the cube clockwise."""
        self.state[int(face)] = np.rot90(self.state[int(face)], 1)
        match face:
            case SquareColor.WHITE:
                # Rotate the adjacent faces
                row_f_1 = self.get_face_line(1, 0, None).copy()
                row_f_2 = self.get_face_line(2, 0, None).copy()
                row_f_4 = self.get_face_line(4, 0, None).copy()
                row_f_5 = self.get_face_line(5, 0, None).copy()
                self.__set_face_line(1, 0, None, row_f_5)
                self.__set_face_line(2, 0, None, row_f_1)
                self.__set_face_line(4, 0, None, row_f_2)
                self.__set_face_line(5, 0, None, row_f_4)
            case SquareColor.RED:
                # Rotate the adjacent faces
                col_f_0 = self.get_face_line(0, None, 2, True).copy()
                col_f_2 = self.get_face_line(2, None, 2).copy()
                col_f_3 = self.get_face_line(3, None, 2).copy()
                col_f_5 = self.get_face_line(5, None, 0, True).copy()
                self.__set_face_line(0, None, 2, col_f_2)
                self.__set_face_line(2, None, 2, col_f_3)
                self.__set_face_line(3, None, 2, col_f_5)
                self.__set_face_line(5, None, 0, col_f_0)
            case SquareColor.GREEN:
                # Rotate the adjacent faces
                row_f_0 = self.get_face_line(0, 2, None, True).copy()
                col_f_1 = self.get_face_line(1, None, 0).copy()
                row_f_3 = self.get_face_line(3, 0, None, True).copy()
                col_f_4 = self.get_face_line(4, None, 2).copy()
                self.__set_face_line(0, 2, None, col_f_4)
                self.__set_face_line(1, None, 0, row_f_0)
                self.__set_face_line(3, 0, None, col_f_1)
                self.__set_face_line(4, None, 2, row_f_3)
            case SquareColor.YELLOW:
                # Rotate the adjacent faces
                row_f_1 = self.get_face_line(1, 2, None).copy()
                row_f_2 = self.get_face_line(2, 2, None).copy()
                row_f_4 = self.get_face_line(4, 2, None).copy()
                row_f_5 = self.get_face_line(5, 2, None).copy()
                self.__set_face_line(1, 2, None, row_f_2)
                self.__set_face_line(2, 2, None, row_f_4)
                self.__set_face_line(4, 2, None, row_f_5)
                self.__set_face_line(5, 2, None, row_f_1)
            case SquareColor.ORANGE:
                # Rotate the adjacent faces
                col_f_0 = self.get_face_line(0, None, 0).copy()
                col_f_2 = self.get_face_line(2, None, 0).copy()
                col_f_3 = self.get_face_line(3, None, 0, True).copy()
                col_f_5 = self.get_face_line(5, None, 2, True).copy()
                self.__set_face_line(0, None, 0, col_f_5)
                self.__set_face_line(2, None, 0, col_f_0)
                self.__set_face_line(3, None, 0, col_f_2)
                self.__set_face_line(5, None, 2, col_f_3)
            case SquareColor.BLUE:
                # Rotate the adjacent faces
                row_f_0 = self.get_face_line(0, 0, None, True).copy()
                col_f_1 = self.get_face_line(1, None, 2).copy()
                row_f_3 = self.get_face_line(3, 2, None).copy()
                col_f_4 = self.get_face_line(4, None, 0, True).copy()
                self.__set_face_line(0, 0, None, col_f_1)
                self.__set_face_line(1, None, 2, row_f_3)
                self.__set_face_line(3, 2, None, col_f_4)
                self.__set_face_line(4, None, 0, row_f_0)
        return self
    def rotate_counter_clockwise_once(self, face: SquareColor) -> Self:
        """Rotates the face of the cube counterclockwise."""
        return self.rotate_clockwise_once(face).rotate_clockwise_once(face).rotate_clockwise_once(face)
    def rotate_twice(self, face: SquareColor) -> Self:
        """Rotates the face of the cube twice."""
        return self.rotate_clockwise_once(face).rotate_clockwise_once(face)
