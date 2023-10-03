# BEGIN: 7z8f3j5d9x4c
import filtering
import cv2 as cv
import numpy as np
import os

images = []
for filename in os.listdir("images"):
    if filename.startswith("cube"):
        img = cv.imread(os.path.join("images", filename))
        if img is not None:
            images.append(img)

# Resize all images to 800x600
images = [cv.resize(img, (800, 600)) for img in images]
images_1 = np.vstack(images)
# Apply filter to all images
images_filtered = [filtering.amax_filter(img) for img in images]
#Convert images to BGR
images_filtered = [cv.cvtColor(img, cv.COLOR_GRAY2BGR) for img in images_filtered]
images_2 = np.vstack(images_filtered)
# merge images_1 with images_2 and resize to 800x600
images_gray = [cv.cvtColor(cv.cvtColor(img, cv.COLOR_BGR2GRAY),cv.COLOR_GRAY2BGR) for img in images]
images_3 = np.vstack(images_gray)

result = np.hstack((images_1, images_2, images_3))
result = cv.resize(result, (800, 600))
cv.imshow("Concatenated Image", result)
cv.waitKey(0)
cv.destroyAllWindows()
images_filtered