from typing import List
import numpy as np
import cv2 as cv
import math

def estimate_focal_length(image_width: float, fov: float) -> List[np.ndarray]:
    """Returns an estimate of the camera focal length given the image width and the field of view."""
    return (image_width/2) * (1/math.tan(math.radians(fov/2)))

def build_camera_matrix(fov: float, image_width: float, image_height: float) -> np.ndarray:
    """Returns the camera matrix"""
    focal_length = estimate_focal_length(image_width, fov)
    return np.array([[focal_length, 0, image_width/2], [0, focal_length, image_height/2], [0, 0, 1]])

def estimate_rectangle_contour_pose(contour: np.ndarray, camera_matrix: np.ndarray) -> tuple:
    """Returns the rotation and translation vectors of the contour pose, using the camera matrix and the image size."""
    #Assume the contour comes directly from the process_frame pipeline so the last dimensions need to be removed
    rect_points_2d = contour.squeeze(axis=1).astype(np.float32)
    # If the contour is not a rectangle, then we need to find the minimum area rectangle
    if len(contour) != 4:
        contour = np.int0(cv.boxPoints(cv.minAreaRect(contour)))
        rect_points_2d = contour.astype(np.float32)
    rect_points_3d = np.array([(-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0)])
    (success, rotation_vector, translation_vector) = cv.solvePnP(rect_points_3d, rect_points_2d, camera_matrix, None, flags=cv.SOLVEPNP_ITERATIVE)
    assert success, "solvePnP failed to find the pose of the contour"
    return (rotation_vector, translation_vector)