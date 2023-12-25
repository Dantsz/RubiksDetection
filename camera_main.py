import numpy as np
import cv2 as cv
from python_scripts import viewport_properties

def camera_main_loop(frame_process_routine):
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # Load an image
        img = frame
        img = cv.resize(img, (500, 400))
        # Apply the routine
        frame_process_routine(img)

def camera_main_coroutine(frame_process_routine):
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # Load an image
        img = frame
        img = cv.resize(img, (viewport_properties.WIDTH, viewport_properties.HEIGHT))
        # Apply the routine
        frame_process_routine(img)
        yield