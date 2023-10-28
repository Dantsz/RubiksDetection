import cv2 as cv
import numpy as np

def display_lines_houghP(img: cv.Mat,lines) -> cv.Mat:
    if lines is None:
        return img
    ret = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    # Draw the detected lines on the original image
    for line in lines:
        x1, y1, x2, y2 = line
        cv.line(ret, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return ret

def display_intersection_points(img: cv.Mat, intersection_points: [np.ndarray]) -> cv.Mat:
    for point in intersection_points:
        if point is not None:
            print(point)
            cv.circle(img, tuple([int(point[0]),int(point[1])]), 5, (0, 255, 0), -1)
    return img