"""Color estimation for Rubik's Cube faces.
"""

from enum import IntEnum
from typing import Tuple
import cv2 as cv
import numpy as np

class SquareColor(IntEnum):
    """Enum to represent the color of a square in a face of the cube."""
    WHITE = 0
    YELLOW = 1
    BLUE = 2
    GREEN = 3
    RED = 4
    ORANGE = 5
    Unknown = 6

# LAB values for the standard rubik's cube colors
# Scaled 0-255 because opencv
reference_colors = np.array([
    [127, 127],
    [110, 185],
    [127, 100],
    [80, 155],
    [160, 146],
    [200, 178],
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
