"""Color estimation for Rubik's Cube faces.
"""

import cv2 as cv
import numpy as np

def color_average_hue(img: np.ndarray) -> float:
    '''Computes the average hue of a contour in an image.'''
    #Convert the image to HSV
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    #Compute the average hue
    return cv.mean(hsv)[0]