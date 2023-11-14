import cv2 as cv
import numpy as np

# """
# FUNCTIONS SHOULD HAVE THE SAME BEHAVIOR:
# TRANFORM THE RGB INTO A BINARY IMAGE WHERE
# THE CUBE SQUARES ARE WHITE AND CUBE LINES ARE BLACK
# """

def adaptive_amax_filter(img: cv.Mat) -> cv.Mat:
    gray = np.amax(img, axis=2)
    #blur
    gray = cv.GaussianBlur(gray, (11, 11), 0)
    gray = cv.medianBlur(gray, 11)
    # Do adaptive thesholding
    thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 31, 2)
    #Do close
    close_kernel = np.ones((7,7), np.uint8)
    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, close_kernel)
    #Do opening
    kernel = np.ones((7, 7), np.uint8)
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
    gray = cv.GaussianBlur(gray, (5, 5), 0)
    edges = cv.Canny(gray, 100, 200)
    return edges

def canny_amax_adaptive_filter(img: cv.Mat) -> cv.Mat:
    gray =  np.amax(img, axis=2)
    gray = cv.GaussianBlur(gray, (11, 11), 0)
    gray = cv.medianBlur(gray, 11)
    # Do adaptive thesholding
    thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 31, 2)
    #Do close
    close_kernel = np.ones((3,3), np.uint8)
    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, close_kernel)
    #Do opening
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
    thresh = cv.GaussianBlur(thresh, (11, 11), 0)
    edges = cv.Canny(thresh, 100, 200)
    return edges

def sobel_amax_filter(img: cv.Mat) -> cv.Mat:
    gray = np.amax(img, axis=2)
    #blur
    gray = cv.GaussianBlur(gray, (15, 15), 0)
    # Apply Sobel filter
    sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=5)
    sobely = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=5)
    sobel = cv.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
    # Convert back to uint8
    sobel = np.uint8(np.absolute(sobel))
    return sobel

def sobel_grayscale_filter(img: cv.Mat) -> cv.Mat:
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #blur
    gray = cv.GaussianBlur(gray, (15, 15), 0)
    # Apply Sobel filter
    sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=5)
    sobely = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=5)
    sobel = cv.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
    # Convert back to uint8
    sobel = np.uint8(np.absolute(sobel))
    return sobel
