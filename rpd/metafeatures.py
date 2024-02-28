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
    avg_hue: float

class PreProcessingData:
    """Structure-of-arrays containing data derived from a contour."""
    centers_of_mass: List[Tuple[int,int] | None]
    areas: List[float | None]
    orientation: List[Tuple[np.ndarray,np.ndarray, np.ndarray] | None]
    normals: List[np.ndarray | None]

    def __init__(self, contours : List[np.ndarray]):
        self.centers_of_mass = []
        self.areas = []
        self.orientation = []
        self.normals = []
        camera_matrix = orientation.build_camera_matrix(vp.ORIENTATION_ESTIMATED_FOV, vp.WIDTH, vp.HEIGHT)
        for contour in contours:
            M = cv.moments(contour)
            if M["m00"] == 0:
                self.centers_of_mass.append(None)
                self.orientation.append(None)
                self.areas.append(None)
                self.normals.append(None)
                continue
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            self.centers_of_mass.append(center)

            (rotation_vector, translation_vector) = orientation.estimate_rectangle_contour_pose(contour, camera_matrix)
            normal_vector = orientation.compute_rectangle_normal_vector(rotation_vector)
            self.orientation.append((rotation_vector, translation_vector, normal_vector))

            self.areas.append(cv.contourArea(contour))
            self.normals.append(normal_vector)
        pass
    def __getitem__(self, index: int) -> Tuple[Tuple[int,int], float, Tuple[np.ndarray,np.ndarray, np.ndarray], np.ndarray] | None:
        if (self.centers_of_mass is None or self.areas is None or self.orientation is None or self.normals is None):
            return None
        return (self.centers_of_mass[index], self.areas[index], self.orientation[index], self.normals[index])

def assemble_face_data(frame, contours: List[np.ndarray], contours_data : PreProcessingData, face_ids: List[int], relative_positions: List[Tuple[float,float]]) -> List[List[FaceSquare]]:
    """Assemble the face data from the preprocessed data."""
    squares: List[FaceSquare] = []
    for idx, id in enumerate(face_ids):
        assert contours_data[id] is not None, f"Contour {id} is None"
        center, area, orientation, normal = contours_data[id]
        square_img = features.contours_crop_and_reverse_perspective(frame, [contours[id]], (100,100))
        avg_hue = color.color_average_hue(square_img[0])
        square = FaceSquare(id, contours[id], center, relative_positions[idx], avg_hue)
        squares.append(square)
    squares = sorted(squares, key=lambda k: k.relative_position[0])
    rows = [sorted(squares[x:x+3],key= lambda k: k.relative_position[1]) for x in range(0, len(squares), 3)]
    return rows

def detect_face(frame, contours: List[np.ndarray]) -> List[List[FaceSquare]] | None:
    """Find the face of the cube in the contours list."""

    assert len(frame.shape) == 3, "The image must be in BGR format, "
    if len(contours) < 9: # We need at least 9 contours to form a cube for now
        return None
    camera_matrix = orientation.build_camera_matrix(vp.ORIENTATION_ESTIMATED_FOV, vp.WIDTH, vp.HEIGHT)
    contours_data = PreProcessingData(contours)
    # Check each contour and find 8 neighbors
    for i in range(len(contours)):
        if contours_data.orientation[i] is None:
            continue
        face = [contours[i]]
        ids = [i]
        (rotation_vector, translation_vector, normal_vector) = contours_data.orientation[i]
        relative_positions = [(features.contour_basis_change(contours_data.centers_of_mass[i], contours_data.centers_of_mass[i], camera_matrix, rotation_vector, translation_vector))]
        centers = [contours_data.centers_of_mass[i]]
        #Find contour orientation
        for j in range(len(contours)):
            if i == j or contours_data.orientation[j] is None:
                continue
            # Find the distance between the centers of mass
            distance = np.sqrt((contours_data.centers_of_mass[i][0] - contours_data.centers_of_mass[j][0])**2 + (contours_data.centers_of_mass[i][1] - contours_data.centers_of_mass[j][1])**2)
            # If the distance is less than 1/3 of the perimeter of the contours, then they are neighbors
            if distance < (cv.arcLength(contours[i], True)/3 + cv.arcLength(contours[j], True)/3) and features.contorus_area_similarity(contours[i], contours[j], 0.025):
                #Find position of the neighbor contour relative to the center contour
                face.append(contours[j])
                ids.append(j)
                relative_positions.append(features.contour_basis_change(contours_data.centers_of_mass[j], contours_data.centers_of_mass[i], camera_matrix, rotation_vector, translation_vector))
                centers.append(contours_data.centers_of_mass[j])
        if len(face) == 9:
            rows = assemble_face_data(frame, contours, contours_data, ids, relative_positions)
            # Could also skip
            assert (rows[1][1].id == i), f"The center square is not in the center, something went wrong!, got : {rows[1][1].id} expected {i}"
            # TODO: Add other checks for cube face
            return rows
        pass
    return None