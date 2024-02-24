from typing import List, Tuple
import cv2 as cv
import numpy as np

from . import viewport_properties as vp
from . import orientation

def detect_face(contours: List[np.ndarray]) -> List[np.ndarray] | None:
    """Find the face of the cube in the contours list."""
    if len(contours) < 9: # We need at least 9 contours to form a cube for now
        return None
    camera_matrix = orientation.build_camera_matrix(vp.ORIENTATION_ESTIMATED_FOV, vp.WIDTH, vp.HEIGHT)
    contours_orientations: List[Tuple[np.ndarray, np.ndarray]]= []
    # Find the orientation of each contour
    for contour in contours:
        (rotation_vector, _) = orientation.estimate_rectangle_contour_pose(contour, camera_matrix)
        normal_vector = orientation.compute_rectangle_normal_vector(rotation_vector)
        contours_orientations.append((contour, normal_vector))
    # Find the center of mass of each contour
    centers_of_mass = []
    for contour, _ in contours_orientations:
        M = cv.moments(contour)
        if M["m00"] == 0:
            centers_of_mass.append(None)
            continue
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        centers_of_mass.append(center)
    # Check each contour and find 8 neighbors
    for i in range(len(contours)):
        #Find contour orientation
        for j in range(len(contours)):
            if i == j:
                continue
        pass
    return None