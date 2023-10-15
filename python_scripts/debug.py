import cv2 as cv
import numpy as np

def display_lines_houghP(img: cv.Mat,lines) -> cv.Mat:
    if lines is None:
        return img
    ret = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    # Draw the detected lines on the original image
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(ret, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return ret