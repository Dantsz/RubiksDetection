import logging
import math
import numpy as np
import cv2 as cv

from . import filtering
from . import features
from . import viewport_properties
from . import orientation
from . import metafeatures
from . import color
class DetectionEngine:
    def __init__(self):
        logging.info("initializing DetectionEngine")
        pass

    def process_frame(self, frame: np.ndarray):
        img = frame
        assert(frame is not None, "frame is None")
        assert(frame.shape[0] == viewport_properties.HEIGHT, f"{frame.shape[0]} != {viewport_properties.HEIGHT}")
        assert(frame.shape[1] == viewport_properties.WIDTH, f"{frame.shape[1]} != {viewport_properties.WIDTH}")
        assert(frame.shape[2] == 3, f"{frame.shape[2]} != 3, the image must be in BGR format")
        frame = filtering.canny_amax_adaptive_filter(frame)
        contours, hierarchy = cv.findContours(frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = features.contours_filter_small_area(contours, viewport_properties.FEATURES_FILTER_MIN_AREA)
        contours = features.contours_filter_large(contours, viewport_properties.FEATURES_FILTER_MAX_AREA)
        contours = features.contours_filter_solidity(contours, viewport_properties.FEATURES_FILTER_SOLIDITY)
        contours = features.contours_filter_isolated_contours(contours, viewport_properties.FEATURES_FILTER_POSITIONAL_2_DISTANCE)
        contours = features.approx_polygon_from_contour(contours)
        # contours = features.contours_min_area_rect(contours)
        self.last_frame = frame
        self.last_contours = contours
        self.last_face = metafeatures.detect_face(img, contours)

    def debug_frame(self, frame: np.ndarray) -> np.ndarray:
        '''Draws debug info on the frame, if it's none it will be draw on a black image'''
        def draw(img, center, imgpts):
            p1 = (int(imgpts[0][0]), int(imgpts[0][1]))
            p2 = (int(imgpts[1][0]), int(imgpts[1][1]))
            p3 = (int(imgpts[2][0]), int(imgpts[2][1]))
            img = cv.line(img, center, p1, (255,0,0), 5)
            img = cv.line(img, center, p2, (0,255,0), 5)
            img = cv.line(img, center, p3, (0,0,255), 5)
            return img
        if frame is None:
            frame = np.zeros((viewport_properties.HEIGHT, viewport_properties.WIDTH, 3), np.uint8)

        img_2 = cv.drawContours(frame, self.last_contours, -1, (0,255,0), 3)
        for contour in self.last_contours:
            M = cv.moments(contour)
            if M["m00"] == 0:
                continue
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            # Use an estimated fov of 65 degrees
            fov = viewport_properties.ORIENTATION_ESTIMATED_FOV
            camera_matrix = orientation.build_camera_matrix(fov, viewport_properties.WIDTH, viewport_properties.HEIGHT)
            (rotation_vector, translation_vector) = orientation.estimate_rectangle_contour_pose(contour, camera_matrix)
            axis = np.float32([[1,0,0], [0,1,0], [0,0,-1]]).reshape(-1,3)
            imgpts, jac = cv.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, None)
            imgpts = imgpts.squeeze(axis=1)
            # img_2 = draw(img_2, center, imgpts)
            # cv.putText(img_2, f'{cv.contourArea(contour)}', (center[0], center[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)
        face = self.last_face
        if face is None:
            print("No face detected")
        else:
            for i, row in enumerate(face):
                for j, square in enumerate(row):
                    img_2 = cv.drawContours(img_2, [square.contour], -1, (0,0,255), 3)
                    # cv.putText(img_2, f'{(i,j)}', square.center, cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)
                    cv.putText(img_2, f'{int(color.color_hue_metric(square.avg_hue))}', square.center, cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)
                    # cv.putText(img_2, f'{cv.contourArea(square.contour)}', square.center, cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)
        return img_2
