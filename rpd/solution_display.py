import numpy as np
from enum import Enum
import logging
import copy

import cv2 as cv

from RubiksDetection.rpd.color import SquareColor
from RubiksDetection.rpd.cube_state import CubeState
from . import metafeatures

class DisplaySolutionResult(Enum):
    GOT_FACE = 0 # Means face is correct
    FAILED_FACE = 1 # Means face is not correct and direction to the face should be drawn
    DONE = 2 # No more moves to display
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
        # helper info
        self.first_tick = True# Will be true before and during the first tick of the current solution
        self.first_solved_tick = True # If the tick of display state is before the second time display_state has been called on a state
        # hooks
        def empty():
            pass
        self.on_initialize = empty
        self.on_solution_start = empty
        self.on_solution_done = empty

    def consume_solution(self, detected_centers, state: CubeState, solving_moves: list[str]):
        self.on_initialize()
        self.centers = detected_centers
        self.solving_moves = solving_moves
        self.cube_state = state
        self.remaining_moves = solving_moves.copy()

    def ready(self):
        return self.centers is not None and self.solving_moves is not None

    def __move_code_to_face(self, move: str) -> SquareColor:
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

    def __move_str_to_face_and_direction(self, move: str) -> tuple[SquareColor, int]:
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

    def __valid_target_face(self, target_face: SquareColor, current_state: CubeState, expected_state: CubeState) -> bool:
        """If after the move the target face is the same it cannot be a target face"""
        return not np.array_equal(current_state.get_face(target_face), expected_state.get_face(target_face))

    def __get_move_display_target_face(self, move: str, current_state: CubeState, expected_state: CubeState) -> tuple[SquareColor, SolutionDisplayRelativeLocation]:
        """Return the face the move should be displayed on."""
        color, _ = self.__move_str_to_face_and_direction(move)
        match color:
            case SquareColor.WHITE:
                if not self.__valid_target_face(SquareColor.GREEN, current_state, expected_state):
                    # logging.warning(f"Move {move} should not be displayed on another face")
                    return SquareColor.ORANGE, SolutionDisplayRelativeLocation.TOP
                return SquareColor.GREEN, SolutionDisplayRelativeLocation.TOP
            case SquareColor.RED:
                if not self.__valid_target_face(SquareColor.GREEN, current_state, expected_state):
                    logging.debug(f"Move {move} should not be displayed on another face")
                    return SquareColor.WHITE, SolutionDisplayRelativeLocation.RIGHT
                return SquareColor.GREEN, SolutionDisplayRelativeLocation.RIGHT
            case SquareColor.ORANGE:
                if not self.__valid_target_face(SquareColor.GREEN, current_state, expected_state):
                    logging.debug(f"Move {move} should not be displayed on another face")
                    return SquareColor.WHITE, SolutionDisplayRelativeLocation.LEFT
                return SquareColor.GREEN, SolutionDisplayRelativeLocation.LEFT
            case SquareColor.YELLOW:
                if not self.__valid_target_face(SquareColor.GREEN, current_state, expected_state):
                    logging.debug(f"Move {move} should not be displayed on another face")
                    return SquareColor.ORANGE, SolutionDisplayRelativeLocation.BOTTOM
                return SquareColor.GREEN, SolutionDisplayRelativeLocation.BOTTOM
            case SquareColor.GREEN:
                if not self.__valid_target_face(SquareColor.WHITE, current_state, expected_state):
                    logging.debug(f"Move {move} should not be displayed on another face")
                    return SquareColor.ORANGE, SolutionDisplayRelativeLocation.RIGHT
                return SquareColor.WHITE, SolutionDisplayRelativeLocation.BOTTOM
            case SquareColor.BLUE:
                if not self.__valid_target_face(SquareColor.WHITE, current_state, expected_state):
                    logging.debug(f"Move {move} should not be displayed on another face")
                    return SquareColor.ORANGE, SolutionDisplayRelativeLocation.LEFT
                return SquareColor.WHITE, SolutionDisplayRelativeLocation.TOP

    def __classify_face_squares(self, face: metafeatures.Face) -> np.ndarray:
        classified_face = np.zeros((3, 3), dtype=int)
        for i, row in enumerate(face):
            for j, square in enumerate(row):
                color = 6
                cur_best = None
                for k, center in enumerate(self.centers):
                    dist = np.linalg.norm(center - square.avg_lab)
                    if cur_best is None or dist < cur_best:
                        cur_best = dist
                        color = k
                classified_face[i, j] = color
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

    def __draw_color_line(self, frame: np.ndarray, color: tuple[int, int, int], start: tuple[int, int], end: tuple[int, int], direction: int) -> np.ndarray:
        if direction == 1:
            cv.arrowedLine(frame, start, end, color, 5)
        elif direction == -1:
            cv.arrowedLine(frame, end, start, color, 5)
        elif direction == 2:
            cv.line(frame, start, end, color, 5)
        return frame

    def draw_move(self, frame: np.ndarray, move: str, face: metafeatures.Face, location: SolutionDisplayRelativeLocation) -> np.ndarray:
        """Draws the move on the face."""
        _, direction = self.__move_str_to_face_and_direction(move)
        start, end = None, None
        match location:
            case SolutionDisplayRelativeLocation.LEFT:
                start = face.faces[0][0].center
                end = face.faces[0][2].center
            case SolutionDisplayRelativeLocation.RIGHT:
                start = face.faces[2][2].center
                end = face.faces[2][0].center
            case SolutionDisplayRelativeLocation.TOP:
                start = face.faces[2][0].center
                end = face.faces[0][0].center
            case SolutionDisplayRelativeLocation.BOTTOM:
                start = face.faces[0][2].center
                end = face.faces[2][2].center
        frame = self.__draw_color_line(frame, (0, 255, 0), start, end, direction)
        return frame

    def __draw_text_above_face(self, frame: np.ndarray, face: metafeatures.Face, text: str) -> np.ndarray:
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (0, 255, 0)
        line_type = 2
        middle_up_center = face.faces[1][0].center
        middle_center = face.faces[1][1].center
        text_size = cv.getTextSize(text, font, font_scale, line_type)[0]
        text_x =  text_size[0] // 2
        text_y =  text_size[1] // 2

        display_vec = (middle_up_center[0] + middle_up_center[0] - middle_center[0], middle_up_center[1] + middle_up_center[1] - middle_center[1])
        cv.putText(frame, text, (display_vec[0] - text_x, display_vec[1]), font, font_scale, font_color, line_type)
        return frame

    def __draw_face_change_move(self, frame: np.ndarray, target_face: SquareColor, face : metafeatures.Face) -> np.ndarray:
        """Draws the text to indicate the face change."""
        text = f"Change face to: {target_face.name}"
        return self.__draw_text_above_face(frame, face, text)

    def display_solution(self, frame: np.ndarray, face : metafeatures.Face) -> tuple[np.ndarray, DisplaySolutionResult]:
        # logging.info(f"Current face {self.__classify_face_squares(face)}")
        if self.first_tick:
            self.on_solution_start()
            self.first_tick = False

        if len(self.remaining_moves) == 0:
            frame = self.__draw_text_above_face(frame, face, "Solved")
            if self.first_solved_tick:
             self.on_solution_done()
             self.first_solved_tick = False
            return frame, DisplaySolutionResult.DONE
        move = self.remaining_moves[0]
        expected_state: CubeState = self.__get_expected_state_after_move(move)
        target_face, target_location = self.__get_move_display_target_face(move, self.cube_state, expected_state)
        expected_face = expected_state.get_face(target_face)
        detected_face = self.__classify_face_squares(face)
        current_face = self.cube_state.get_face(target_face)

        # print(f"Expected face {expected_face}, after move {move}")

        if expected_face[1][1] == detected_face[1][1]:
            if np.array_equal(expected_face, detected_face):
                self.remaining_moves.pop(0)
                self.cube_state = expected_state
                print(f"Move {move} is correct")
            else:
                frame = self.draw_move(frame, move, face, target_location)
        else:
            print(f"Face {detected_face[1][1]} is incorrect, should be {expected_face[1][1]}")
            frame = self.__draw_face_change_move(frame, target_face, face)
            return frame, DisplaySolutionResult.FAILED_FACE

        # logging.info(f"Move is : {move}, should display on color {self.__get_move_display_target_face(move)}")
        # face_contour = face.get_face_contour()
        # frame = cv.drawContours(frame, [face_contour], -1, (255, 0, 255), 2)
        return frame, DisplaySolutionResult.GOT_FACE
        #Flow:
            #Decide which face should be displayed based on the move
            #Detect if the face is the one to perform the move on, else DRAW MOVE TO GET TO THAT FACE
            #If the face is correct draw the move
            #Detect if the face changed as it should have
            #Take the first move and combine it with the face to draw the move
