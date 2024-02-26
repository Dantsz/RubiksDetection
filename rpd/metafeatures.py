import cv2 as cv
import numpy as np

from dataclasses import dataclass
from typing import List, Tuple

from . import viewport_properties as vp
from . import orientation
from . import features
from . import color

@dataclass
class FaceSquare:
    """Dataclass to store the information of a square in a face of the cube. The face is a 3x3 grid of FaceSquares."""
    id: int
    contour: np.ndarray
    center: Tuple[int,int]
    relative_position: Tuple[float,float]

def detect_face(contours: List[np.ndarray]) -> List[List[FaceSquare]] | None:
    """Find the face of the cube in the contours list."""
    if len(contours) < 9: # We need at least 9 contours to form a cube for now
        return None
    camera_matrix = orientation.build_camera_matrix(vp.ORIENTATION_ESTIMATED_FOV, vp.WIDTH, vp.HEIGHT)
    contours_orientations: List[Tuple[np.ndarray,np.ndarray, np.ndarray] | None]= []
    # Find the orientation of each contour
    # Find the center of mass of each contour
    # Find contour area
    centers_of_mass : List[Tuple[int,int] | None]= []
    areas = []
    for contour in contours:
        M = cv.moments(contour)
        if M["m00"] == 0:
            centers_of_mass.append(None)
            contours_orientations.append(None)
            continue
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        centers_of_mass.append(center)

        (rotation_vector, translation_vector) = orientation.estimate_rectangle_contour_pose(contour, camera_matrix)
        normal_vector = orientation.compute_rectangle_normal_vector(rotation_vector)
        contours_orientations.append((rotation_vector, translation_vector, normal_vector))

        areas.append(cv.contourArea(contour))
    # Check each contour and find 8 neighbors
    for i in range(len(contours)):
        if contours_orientations[i] is None:
            continue
        face = [contours[i]]
        ids = [i]
        (rotation_vector, translation_vector, normal_vector) = contours_orientations[i]
        relative_positions = [(features.contour_basis_change(centers_of_mass[i], centers_of_mass[i], camera_matrix, rotation_vector, translation_vector))]
        centers = [centers_of_mass[i]]
        #Find contour orientation
        for j in range(len(contours)):
            if i == j or contours_orientations[j] is None:
                continue
            # Find the distance between the centers of mass
            distance = np.sqrt((centers_of_mass[i][0] - centers_of_mass[j][0])**2 + (centers_of_mass[i][1] - centers_of_mass[j][1])**2)
            # If the distance is less than 1/3 of the perimeter of the contours, then they are neighbors
            if distance < (cv.arcLength(contours[i], True)/3 + cv.arcLength(contours[j], True)/3) and features.contorus_area_similarity(contours[i], contours[j], 0.1):
                #Find position of the neighbor contour relative to the center contour
                face.append(contours[j])
                ids.append(j)
                relative_positions.append(features.contour_basis_change(centers_of_mass[j], centers_of_mass[i], camera_matrix, rotation_vector, translation_vector))
                centers.append(centers_of_mass[j])
        if len(face) == 9:
            # Create list of dictionaries
            # Sort into rows and columns by relative position
            squares :List[FaceSquare] = []
            for x in range(len(face)):
                square = FaceSquare(ids[x], face[x], centers[x], relative_positions[x])
                squares.append(square)
            squares = sorted(squares, key=lambda k: k.relative_position[0])
            #Split into 3 rows
            rows = [sorted(squares[x:x+3],key= lambda k: k.relative_position[1]) for x in range(0, len(squares), 3)]
            # Could also skip
            assert (rows[1][1].id == i), f"The center square is not in the center, something went wrong!, got : {rows[1][1].id} expected {i}"
            # TODO: Add other checks for cube face
            return rows
        pass
    return None