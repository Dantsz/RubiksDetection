"""Color estimation for Rubik's Cube faces.
"""

from enum import IntEnum
from typing import Tuple
import cv2 as cv
import numpy as np

class SquareColor(IntEnum):
    """Enum to represent the color of a square in a face of the cube."""
    WHITE = 0
    RED = 1
    GREEN = 2
    YELLOW = 3
    ORANGE = 4
    BLUE = 5
    Unknown = 6

# LAB values for the standard rubik's cube colors
# Scaled 0-255 because opencv
reference_colors = np.array([
    [127, 127], # white
    [190, 165], # red
    [80, 155], # green
    [110, 185], # yellow
    [200, 180], # orange
    [127, 100], # blue
])

def color_avg_lab(img: np.ndarray) -> Tuple[float, float, float]:
    '''Computes the average L*a*b* of a contour in an image.'''
    lab = cv.cvtColor(img, cv.COLOR_BGR2Lab)
    mean = cv.mean(lab)
    return mean[0], mean[1], mean[2]

def color_median_lab(img: np.ndarray) -> Tuple[float, float, float]:
    '''Computes the median L*a*b* of a contour in an image.'''
    lab = cv.cvtColor(img, cv.COLOR_BGR2Lab)
    median_channel0 = np.median(lab[:,:,0])
    median_channel1 = np.median(lab[:,:,1])
    median_channel2 = np.median(lab[:,:,2])
    return median_channel0, median_channel1, median_channel2

def color_to_spatial_symbol(color: SquareColor) -> str:
    '''Converts a color to a spatial symbol.'''
    if color == SquareColor.WHITE:
        return 'U'
    elif color == SquareColor.YELLOW:
        return 'D'
    elif color == SquareColor.BLUE:
        return 'B'
    elif color == SquareColor.GREEN:
        return 'F'
    elif color == SquareColor.RED:
        return 'R'
    elif color == SquareColor.ORANGE:
        return 'L'
    else:
        return 'X'

def spatial_symbol_to_color(move_str: str) -> SquareColor:
    '''Converts a spatial symbol to a color.'''
    symbol = move_str[0]
    if symbol == 'U':
        return SquareColor.WHITE
    elif symbol == 'D':
        return SquareColor.YELLOW
    elif symbol == 'B':
        return SquareColor.BLUE
    elif symbol == 'F':
        return SquareColor.GREEN
    elif symbol == 'R':
        return SquareColor.RED
    elif symbol == 'L':
        return SquareColor.ORANGE
    else:
        return SquareColor.Unknown

def  move_code_to_face(move: str) -> SquareColor:
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