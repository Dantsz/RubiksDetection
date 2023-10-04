import numpy as np
import cv2 as cv
import filtering

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
    img_4 = filtering.canny_convert_filter(img)

    images = np.vstack((np.hstack((img_1, img_2)),np.hstack((img_3, img_4))))
    images= cv.resize(images, (1920, 600))

    # Display the image

    # Invert the image
    img_1 = cv.bitwise_not(img_1)
    # Apply Hough transform on the detected edges to detect lines
    lines = cv.HoughLinesP(img_1, rho=1, theta=np.pi/180, threshold=30, minLineLength=100, maxLineGap=2)
    # Convert the image to RGB
    img_1 = cv.cvtColor(img_1, cv.COLOR_GRAY2BGR)
    # Draw the detected lines on the original image
    print(f"Number of lines detected: {len(lines)}")
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(img_1, (x1, y1), (x2, y2), (0, 0, 255), 2)
    # # Resize and display the image
    cv.imshow('Image', images)
    cv.imshow('Lines', img_1)
    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
