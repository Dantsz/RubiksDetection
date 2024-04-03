from dataclasses import asdict
import json
import logging
import math
import pickle
import numpy as np
import cv2 as cv

from . import filtering
from . import features
from . import viewport_properties
from . import orientation
from . import metafeatures
from . import cube_state
class DetectionEngine:
    def __init__(self):
        logging.info("initializing DetectionEngine")
        self.last_face = None
        self.orientation_correction = True
        pass

    def process_frame(self, frame: np.ndarray):
        img = frame
        assert frame is not None, "frame is None"
        assert frame.shape[0] == viewport_properties.HEIGHT, f"{frame.shape[0]} != {viewport_properties.HEIGHT}"
        assert frame.shape[1] == viewport_properties.WIDTH, f"{frame.shape[1]} != {viewport_properties.WIDTH}"
        assert frame.shape[2] == 3, f"{frame.shape[2]} != 3, the image must be in BGR format"
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
        face = metafeatures.detect_face(img, contours, self.orientation_correction)
        if face is not None:
            self.last_face = face


    def debug_frame(self, frame: np.ndarray, draw_orientation: bool = False, draw_contours: bool = True, draw_face = True, draw_avg_color: bool = False, draw_coordinates: bool = False) -> np.ndarray:
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
        img_2 = frame

        if draw_contours:
            img_2 = cv.drawContours(img_2, self.last_contours, -1, (0,255,0), 3)
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
            if draw_orientation:
                img_2 = draw(img_2, center, imgpts)
            # cv.putText(img_2, f'{cv.contourArea(contour)}', (center[0], center[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)
        face = self.last_face
        if face is None:
            pass
        else:
            for i, row in enumerate(face):
                for j, square in enumerate(row):
                    if draw_face:
                        img_2 = cv.drawContours(img_2, [square.contour], -1, (0,0,255), 3)
                    # cv.putText(img_2, f'{(i,j)}', square.center, cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)
                    color_met = square.avg_lab
                    if draw_avg_color:
                        text_size = cv.getTextSize(f'{(int(color_met[0]),int(color_met[1]), int(color_met[2]))}', cv.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
                        text_x =  text_size[0] // 2
                        text_y =  text_size[1] // 2
                        img_2 = cv.putText(img_2, f'{(int(color_met[0]),int(color_met[1]), int(color_met[2]))}', (square.center[0] - text_x, square.center[1]), cv.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv.LINE_AA)
                    if draw_coordinates:
                        coords_1 = square.center
                        coords_2 = square.relative_position
                        text_size = cv.getTextSize(f'{(coords_1[0], coords_1[1])}', cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                        img_2 = cv.putText(img_2, f'{(coords_1[0], coords_1[1])}', (coords_1[0] - text_size[0]//2, coords_1[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv.LINE_AA)
                        img_2 = cv.putText(img_2, f'{(coords_2[0], coords_2[1])}', (coords_1[0] - text_size[0]//2, coords_1[1] + 10), cv.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv.LINE_AA)
                    # cv.putText(img_2, f'{cv.contourArea(square.contour)}', square.center, cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)
                    color = cv.cvtColor(np.array([[color_met]], dtype=np.uint8),cv.COLOR_LAB2BGR)[0][0]
                    color = (float(color[0]), float(color[1]), float(color[2]))
                    rectangle_pos = (i * 25, j * 25)
                    img_2 = cv.rectangle(img_2, rectangle_pos, (rectangle_pos[0] + 25, rectangle_pos[1] + 25), color, -1)
        return img_2
