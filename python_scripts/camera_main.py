import numpy as np
import cv2 as cv
import filtering
import features
import debug

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
    img = cv.resize(img, (800, 600))

    img_1 = filtering.adaptive_amax_filter(img)
    img_2 = filtering.adaptive_high_pass_filter(img)

    img_3 = filtering.canny_convert_filter(img)
    img_4 = filtering.sobel_grayscale_filter(img)

    images = np.vstack((np.hstack((img_1, img_2)),np.hstack((img_3, img_4))))
    images= cv.resize(images, (1920, 600))

    # Invert the image, looking for cube outline
    img_1 = cv.bitwise_not(img_1)
    # Apply Hough transform on the detected edges to detect lines
    lines = features.extract_lines_houghP(img_1)
    lines = features.filter_perpendicular_lines(lines)
    lined = debug.display_lines_houghP(img_1, lines)
    # Resize and display the image
    cv.imshow('Image', images)
    cv.imshow('Lines', lined)
    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
