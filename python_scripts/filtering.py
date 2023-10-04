import cv2 as cv
import numpy as np

def adaptive_amax_filter(img: cv.Mat) -> cv.Mat:
    gray = np.amax(img, axis=2)
    #blur
    gray = cv.GaussianBlur(gray, (11, 11), 0)
    gray = cv.medianBlur(gray, 11)
    # Do adaptive thesholding
    thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 31, 2)
    #Do close
    close_kernel = np.ones((5, 5), np.uint8)
    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, close_kernel)
    #Do opening
    kernel = np.ones((11, 11), np.uint8)
    thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
    return thresh

def adaptive_high_pass_filter(img: cv.Mat) -> cv.Mat:
    gray = np.amax(img, axis=2)
    #blur
    gray = cv.medianBlur(gray, 15)
    gray = cv.GaussianBlur(gray, (5, 5), 0)
    # Apply Laplacian filter
    laplacian = cv.Laplacian(gray, cv.CV_64F,ksize = 5)
    # Convert back to uint
    laplacian = np.uint8(np.absolute(laplacian))
    #Threshold the inmage at 45
    _, thresh = cv.threshold(laplacian, 45, 255, cv.THRESH_BINARY)
    return thresh

def high_pass_grayscale_filter(img: cv.Mat) -> cv.Mat:
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #blur
    gray = cv.medianBlur(gray, 11)
    gray = cv.GaussianBlur(gray, (5, 5), 0)
    # Apply Laplacian filter
    laplacian = cv.Laplacian(gray, cv.CV_64F,ksize = 5)
    # Convert back to uint8
    laplacian = np.uint8(np.absolute(laplacian))
    return laplacian


def canny_convert_filter(img: cv.Mat) -> cv.Mat:
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5, 5), 0)
    edges = cv.Canny(gray, 100, 200)
    return edges

def canny_amax_filter(img: cv.Mat) -> cv.Mat:
    gray =  np.amax(img, axis=2)
    gray = cv.medianBlur(gray, 11)
    edges = cv.Canny(gray, 100, 200)
    return edges


