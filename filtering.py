import cv2 as cv
import numpy as np
from . import viewport_properties as vp
# """
# FUNCTIONS SHOULD HAVE THE SAME BEHAVIOR:
# TRANFORM THE RGB INTO A BINARY IMAGE WHERE
# THE CUBE SQUARES ARE WHITE AND CUBE LINES ARE BLACK
# """

def amax_adaptive_filter(img, gaussian_blur_kerner: int = vp.FILTER_GAUSSIAN_DEFAULT_KSIZE, morphological_kernel_size: int = vp.FILTER_MORPHOLOGICAL_DEFAULT_KSIZE) :
    gray =  np.amax(img, axis=2)
    gray = cv.convertScaleAbs(gray)
    gray = cv.GaussianBlur(gray, (gaussian_blur_kerner, gaussian_blur_kerner), 0)
    # Do adaptive thesholding
    thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, vp.FILTER_ADAPTIVE_THRESHOLD_BLOCK_SIZE, vp.FILTER_ADAPTIVE_THRESHOLD_CONSTANT)
    #Do close
    close_kernel = np.ones((morphological_kernel_size,morphological_kernel_size), np.uint8)
    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, close_kernel)
    #Do opening
    kernel = np.ones((morphological_kernel_size, morphological_kernel_size), np.uint8)
    thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
    thresh = cv.erode(thresh, np.ones((vp.FILTER_MORPHOLOGICAL_DEFAULT_KSIZE, vp.FILTER_MORPHOLOGICAL_DEFAULT_KSIZE), np.uint8), iterations=1)
    return thresh

