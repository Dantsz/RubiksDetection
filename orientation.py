"""Orientation estimaton for rectangles in images.
"""
import numpy as np
import cv2 as cv
import math
from typing import List, Tuple

def estimate_focal_length(image_dim: float, fov: float) -> List[np.ndarray]:
    """Returns an estimate of the camera focal length given the an image dimension and the field of view."""
    return (image_dim/2) * (1/math.tan(math.radians(fov/2)))

def build_camera_matrix(fov: tuple[float, float], image_width: float, image_height: float) -> np.ndarray:
    """Returns the camera matrix"""
    (fov_x, fov_y) = fov
    focal_length_x = estimate_focal_length(image_width, fov_x)
    focal_length_y = estimate_focal_length(image_height, fov_y)
    return np.array([[focal_length_x, 0, image_width/2], [0, focal_length_y, image_height/2], [0, 0, 1]])

def estimate_rectangle_contour_pose(contour: np.ndarray, camera_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Returns the rotation and translation vectors of the contour pose, using the camera matrix and the image size."""
    #Assume the contour comes directly from the process_frame pipeline so the last dimensions need to be removed
    rect_points_2d = contour.squeeze(axis=1).astype(np.float32)
    # If the contour is not a rectangle, then we need to find the minimum area rectangle
    if len(contour) != 4:
        contour = np.int0(cv.boxPoints(cv.minAreaRect(contour)))
        rect_points_2d = contour.astype(np.float32)
    rect_points_3d = np.array([(-1.0, 1.0, 0.0), (1.0, 1.0, 0.0), (1.0, -1.0, 0.0), (-1.0, -1.0, 0.0)])
    (success, rotation_vector, translation_vector) = cv.solvePnP(rect_points_3d, rect_points_2d, camera_matrix, None, flags=cv.SOLVEPNP_IPPE_SQUARE)
    assert success, "solvePnP failed to find the pose of the contour"
    return (rotation_vector, translation_vector)

def compute_rectangle_normal_vector(rotation_vector: np.ndarray) -> np.ndarray:
    """Returns the normal vector of the rectangle plane."""
    rotation_matrix, _ = cv.Rodrigues(rotation_vector)
    return np.dot(rotation_matrix, np.array([0, 0, 1]))