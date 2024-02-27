"""Color estimation for Rubik's Cube faces.
"""

import cv2 as cv
import numpy as np
import math

def color_average_hue(img: np.ndarray) -> float:
    '''Computes the average hue of a contour in an image.'''
    #Convert the image to HSV
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    #Compute the average hue
    return cv.mean(hsv)[0]

def color_hue_metric(hue: float) -> float:
    '''Return a number that can be used for clustering the colors by hue.'''
    return math.cos(math.radians(hue)) * 100