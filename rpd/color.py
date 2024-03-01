"""Color estimation for Rubik's Cube faces.
"""

from typing import Tuple
import cv2 as cv
import numpy as np

def color_avg_lab(img: np.ndarray) -> Tuple[float, float, float]:
    '''Computes the average L*a*b* of a contour in an image.'''
    lab = cv.cvtColor(img, cv.COLOR_BGR2Lab)
    mean = cv.mean(lab)
    return mean[0], mean[1], mean[2]
