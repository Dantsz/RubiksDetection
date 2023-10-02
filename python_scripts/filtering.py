import cv2 as cv
import numpy as np

def adaptive_amax_filtre(img: cv.Mat) -> cv.Mat:
    gray = np.amax(img, axis=2)
    #blur
    gray = cv.GaussianBlur(gray, (3, 3), 0)
    gray = cv.medianBlur(gray, 11)
    # Do adaptive thesholding
    thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 31, 2)
    #Do opening
    close_kernel = np.ones((5, 5), np.uint8)
    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, close_kernel)
    return thresh