import camera_main
import filtering
import features
import debug
import numpy as np
import cv2 as cv

def process_frame(frame):
    img = frame
    img = cv.resize(img, (500, 400))

    img_1 = filtering.adaptive_amax_filter(img)
    img_2 = filtering.adaptive_high_pass_filter(img)

    img_3 = filtering.canny_convert_filter(img)
    img_4 = filtering.canny_amax_filter(img)
    img_5 = filtering.canny_amax_adaptive_filter(img)

    images = np.vstack((np.hstack((img_1, img_2))))
    images= cv.resize(images, (1920, 600))

    # Invert the image, looking for cube outline
    img_1 = cv.bitwise_not(img_1)
    # Apply Hough transform on the detected edges to detect lines
    lines = features.extract_lines_houghP(img_1)
    lined = debug.display_lines_houghP(img_1,lines)

    lined = cv.resize(lined,(800,600))
    # Resize and display the image
    cv.imshow('Image', images)
    cv.imshow("Grayscale canny vs Amax canny vs amax adaptive canny", np.hstack((img_3, img_4, img_5)))
    cv.imshow('Lines', lined)
    if cv.waitKey(1) == ord('q'):
       return

camera_main.camera_main_loop(process_frame)