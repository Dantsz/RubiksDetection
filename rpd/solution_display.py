import numpy as np
from enum import Enum
import logging
import copy

import cv2 as cv

from RubiksDetection.rpd.color import SquareColor
from RubiksDetection.rpd.cube_state import CubeState
from . import metafeatures

class SolutionDisplayRelativeLocation(Enum):
    """Which side of the target face needs to be shifted"""
    LEFT = 0
    RIGHT = 1
    TOP = 2
    BOTTOM = 3

class SolutionDisplayEngine:

    def __init__(self):
        self.reset()

    def reset(self):
        self.centers = None
        self.solving_moves = None
        self.remaining_moves = None
        self.cube_state = None

    def consume_solution(self, detected_centers, state: CubeState, solving_moves: list[str]):
        self.centers = detected_centers
        self.solving_moves = solving_moves
        self.cube_state = state
        self.remaining_moves = solving_moves.copy()

    def ready(self):
        return self.centers is not None and self.solving_moves is not None

    def __move_code_to_face(self, move: str) -> metafeatures.Face:
        """Perform a move on a face."""
        assert len(move) == 1, f"Invalid move {move}"
        match move:
            case 'W':
                return SquareColor.WHITE
            case 'R':
                return SquareColor.RED
            case 'G':
                return SquareColor.GREEN
            case 'Y':
                return SquareColor.YELLOW
            case 'O':
                return SquareColor.ORANGE
            case 'B':
                return SquareColor.BLUE
            case _:
                raise ValueError(f"Invalid move {move}")

    def __move_str_to_face_and_direction(self, move: str) -> tuple[metafeatures.Face, int]:
        """Return the face and direction of the move.

        The direction is 1 for clockwise and -1 for counterclockwise and 2 for 180 degrees.
        """
        assert len(move) <= 2, f"Invalid move {move}"
        color = self.__move_code_to_face(move[0])
        if len(move) == 1:
            return color, 1
        if move[1] == "'":
            return color, -1
        if move[1] == "2":
            return color, 2

    def __get_move_display_target_face(self, move: str) -> tuple[SquareColor, SolutionDisplayRelativeLocation]:
        """Return the face the move should be displayed on."""
        color, _ = self.__move_str_to_face_and_direction(move)
        match color:
            case SquareColor.WHITE:
                return SquareColor.GREEN, SolutionDisplayRelativeLocation.TOP
            case SquareColor.RED:
                return SquareColor.GREEN, SolutionDisplayRelativeLocation.RIGHT
            case SquareColor.ORANGE:
                return SquareColor.GREEN, SolutionDisplayRelativeLocation.LEFT
            case SquareColor.YELLOW:
                return SquareColor.GREEN, SolutionDisplayRelativeLocation.BOTTOM
            case SquareColor.GREEN:
                return SquareColor.WHITE, SolutionDisplayRelativeLocation.BOTTOM
            case SquareColor.BLUE:
                return SquareColor.WHITE, SolutionDisplayRelativeLocation.TOP

    def __classify_face_squares(self, face: metafeatures.Face) -> np.ndarray:
        classified_face = np.zeros((3, 3), dtype=SquareColor)
        for i, row in enumerate(face):
            for j, square in enumerate(row):
                color = 6
                cur_best = None
                for k, center in enumerate(self.centers):
                    dist = np.linalg.norm(center - square.avg_lab)
                    if cur_best is None or dist < cur_best:
                        cur_best = dist
                        color = k
                classified_face[j, i] = SquareColor(color)
        return classified_face

    def __get_expected_state_after_move(self, move: str) -> CubeState:
        expected_state = copy.deepcopy(self.cube_state)
        moved_face, direction = self.__move_str_to_face_and_direction(move)
        match direction:
            case 1:
                expected_state.rotate_clockwise_once(moved_face)
            case -1:
                expected_state.rotate_counter_clockwise_once(moved_face)
            case 2:
                expected_state.rotate_twice(moved_face)
        return expected_state



    def display_solution(self, frame: np.ndarray, face : metafeatures.Face) -> np.ndarray:
        # logging.info(f"Current face {self.__classify_face_squares(face)}")
        if len(self.remaining_moves) == 0:
            return frame
        move = self.remaining_moves[0]
        expected_state = self.__get_expected_state_after_move(move)
        target_face, target_location = self.__get_move_display_target_face(move)
        expected_face = expected_state.get_face(target_face)
        print(f"Expected face {expected_face}, after move {move}")

        # logging.info(f"Move is : {move}, should display on color {self.__get_move_display_target_face(move)}")
        # face_contour = face.get_face_contour()
        # frame = cv.drawContours(frame, [face_contour], -1, (255, 0, 255), 2)

        return frame
        #Flow:
            #Decide which face should be displayed based on the move
            #Detect if the face is the one to perform the move on, else DRAW MOVE TO GET TO THAT FACE
            #If the face is correct draw the move
            #Detect if the face changed as it should have
            #Take the first move and combine it with the face to draw the move
