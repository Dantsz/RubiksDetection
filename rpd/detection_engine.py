import logging
import numpy as np
import cv2 as cv

from . import filtering
from . import features
from . import viewport_properties

class DetectionEngine:
    def __init__(self):
        logging.info("initializing DetectionEngine")
        pass

    def process_frame(self, frame: np.ndarray):
        assert(frame is not None, "frame is None")
        assert(frame.shape[0] == viewport_properties.HEIGHT, f"{frame.shape[0]} != {viewport_properties.HEIGHT}")
        assert(frame.shape[1] == viewport_properties.WIDTH, f"{frame.shape[1]} != {viewport_properties.WIDTH}")
        assert(frame.shape[2] == 3, f"{frame.shape[2]} != 3, the image must be in BGR format")
        frame = filtering.canny_amax_adaptive_filter(frame)
        contours, hierarchy = cv.findContours(frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = features.contours_filter_small_area(contours, viewport_properties.FEATURES_FILTER_MIN_AREA)
        contours = features.contours_filter_solidity(contours, viewport_properties.FEATURES_FILTER_SOLIDITY)
        contours = features.contours_filter_isolated_contours(contours, viewport_properties.FEATURES_FILTER_POSITIONAL_2_DISTANCE)
        contours = features.approx_polygon_from_contour(contours)
        self.last_frame = frame
        self.last_contours = contours

    def debug_frame(self, frame: np.ndarray) -> np.ndarray:
        '''Draws debug info on the frame, if it's none it will be draw on a black image'''
        if frame is None:
            frame = np.zeros((viewport_properties.HEIGHT, viewport_properties.WIDTH, 3), np.uint8)

        img_2 = cv.drawContours(frame, self.last_contours, -1, (0,255,0), 3)
        for contour in self.last_contours:
            M = cv.moments(contour)
            if M["m00"] == 0:
                continue
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            cv.putText(img_2, str(len(contour)), center, cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return img_2