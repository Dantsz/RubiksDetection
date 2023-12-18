import cv2 as cv
import numpy as np

# """
# FUNCTIONS SHOULD HAVE THE SAME BEHAVIOR:
# TRANFORM THE RGB INTO A BINARY IMAGE WHERE
# THE CUBE SQUARES ARE WHITE AND CUBE LINES ARE BLACK
# """

def adaptive_amax_filter(img: cv.Mat, gaussian_blur_kerner: int = 7, median_blur_kernel: int = 7, morphological_kernel: int = 5) -> cv.Mat:
    gray = np.amax(img, axis=2)
    #blur
    gray = cv.GaussianBlur(gray, (gaussian_blur_kerner), 0)
    gray = cv.medianBlur(gray, median_blur_kernel)
    # Do adaptive thesholding
    thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 31, 2)
    #Do close
    close_kernel = np.ones((morphological_kernel, morphological_kernel), np.uint8)
    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, close_kernel)
    #Do opening
    kernel = np.ones((morphological_kernel,morphological_kernel), np.uint8)
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


def canny_convert_filter(img: cv.Mat,gaussian_blur_kerner: int = 11, laplacian_k_size: int = 5, dilate_k_size: int = 3) -> cv.Mat:
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.Laplacian(gray, cv.CV_64F,ksize = laplacian_k_size)
    gray = cv.convertScaleAbs(gray)
    gray = cv.dilate(gray, np.ones((dilate_k_size, dilate_k_size), np.uint8), iterations=1)
    gray = cv.GaussianBlur(gray, (gaussian_blur_kerner, gaussian_blur_kerner), 0)
    edges = cv.Canny(gray, 100, 200)
    return edges

def canny_amax_filter(img: cv.Mat) -> cv.Mat:
    gray =  np.amax(img, axis=2)
    gray = cv.GaussianBlur(gray, (5, 5), 0)
    edges = cv.Canny(gray, 100, 200)
    return edges

def canny_amax_adaptive_filter(img: cv.Mat, gaussian_blur_kerner: int = 11, morphological_kernel: int = 3) -> cv.Mat:
    gray =  np.amax(img, axis=2)
    gray = cv.GaussianBlur(gray, (gaussian_blur_kerner, gaussian_blur_kerner), 0)
    # Do adaptive thesholding
    thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 31, 2)
    #Do close
    close_kernel = np.ones((morphological_kernel,morphological_kernel), np.uint8)
    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, close_kernel)
    #Do opening
    kernel = np.ones((morphological_kernel, morphological_kernel), np.uint8)
    thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
    thresh = cv.GaussianBlur(thresh, (gaussian_blur_kerner, gaussian_blur_kerner), 0)
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
