import cv2 as cv
import numpy as np

def extract_lines_houghP(img: cv.Mat):
    lines = cv.HoughLinesP(img, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=5)
    return lines
