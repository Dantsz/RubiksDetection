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

def color_median_lab(img: np.ndarray) -> Tuple[float, float, float]:
    '''Computes the median L*a*b* of a contour in an image.'''
    lab = cv.cvtColor(img, cv.COLOR_BGR2Lab)
    median_channel0 = np.median(lab[:,:,0])
    median_channel1 = np.median(lab[:,:,1])
    median_channel2 = np.median(lab[:,:,2])
    return median_channel0, median_channel1, median_channel2
