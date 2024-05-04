from enum import Enum
import cv2 as cv
import numpy as np
import math
import json

from dataclasses import asdict, dataclass
from typing import Callable, List, Tuple, Union
from numbers import Number

from . import viewport_properties as vp
from . import orientation
from . import features
from . import color

@dataclass
class FaceSquare:
    """Dataclass to store the information of a square in a face of the cube.

    The face is a 3x3 grid of FaceSquares.
    id - index of the square in the contour array
    contour - array of viewport points
    center - center of mass of the contour
    relative_position - position after adjusting the rotation of the square
    avg_lab - average LAB valeus from of the contour
    color - enum representing the color, detect_face does set it to Unknown
    """

    id: int
    contour: np.ndarray
    center: Tuple[int, int]
    relative_position: Tuple[float, float]
    avg_lab: Tuple[float, float, float]

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, dict):
        dict['contour'] = np.array(dict['contour'])
        return cls(**dict)

@dataclass
class Face:
    """Dataclass to store the information of a face of the cube."""
    faces: List[List[FaceSquare]]

    def __getitem__(self, index: Union[int, Tuple[int, int]]) -> FaceSquare:
        if isinstance(index, tuple):
            return self.faces[index[0]][index[1]]
        else:
            return self.faces[index]

    def get_face_contour(self) -> np.ndarray:
        """Return the contour of the face as a list of points."""
        #TODO: optimize this to not use all the contours
        big_contour = np.concatenate([square.contour for row in self.faces for square in row])
        rect = cv.minAreaRect(big_contour)
        box = cv.boxPoints(rect)
        return np.int0(box)

class PreProcessingData:
    """Structure-of-arrays containing data derived from a contour."""

    centers_of_mass: List[Tuple[int, int] | None]
    areas: List[float | None]
    orientation: List[Tuple[np.ndarray, np.ndarray, np.ndarray] | None]
    normals: List[np.ndarray | None]

    def __init__(self, contours: List[np.ndarray]):
        """Build data from detected contours."""
        self.centers_of_mass = []
        self.areas = []
        self.orientation = []
        self.normals = []
        camera_matrix = orientation.build_camera_matrix((vp.ORIENTATION_ESTIMATED_FOV_W, vp.ORIENTATION_ESTIMATED_FOV_H), vp.WIDTH, vp.HEIGHT)
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

    def __getitem__(self, index: int) -> Tuple[Tuple[int, int], float, Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray] | None:
        if (self.centers_of_mass is None or self.areas is None or self.orientation is None or self.normals is None):
            return None
        return (self.centers_of_mass[index], self.areas[index], self.orientation[index], self.normals[index])

def assemble_face_data(frame, contours: List[np.ndarray], contours_data : PreProcessingData, face_ids: List[int], relative_positions: List[Tuple[float, float]], orientation_correction: bool) -> Face | None:
    """Assemble the face data from the preprocessed data."""
    squares: List[FaceSquare] = []
    for idx, id in enumerate(face_ids):
        assert contours_data[id] is not None, f"Contour {id} is None"
        center, area, orientation, normal = contours_data[id]
        square_img = features.contours_crop_and_reverse_perspective(frame, [contours[id]], (100,100))
        avg_lab = color.color_avg_lab(square_img[0])
        square = FaceSquare(id, contours[id], center, relative_positions[idx], avg_lab)
        squares.append(square)

    columns: list[list[FaceSquare]] = __reorder_face(squares, lambda k: k.relative_position)
    # add orientation correction, that means rotate the face so that the top row in the image is the top row of the face
    if orientation_correction:
        columns = correct_face_orientation(columns, squares)
        if columns is None:
            return None

    return Face(columns)

def check_face_integrity(face: Face, center_index) -> bool:
    # TODO: MAKE RETURN FALSE INSTEAD OF ASSERT
    # assert (face[1][1].id == center_index), f"The center square is not in the center, something went wrong!, got : {face[1][1].id} expected {center_index}"
    if (face[1][1].id != center_index):
        return False
    # CHECKS for the angles between the face cross
    def cos_angle(vec_1, vec_2):
        return np.dot(vec_1, vec_2)/(np.linalg.norm(vec_1) * np.linalg.norm(vec_2))
    point_up = np.array(face[0][1].center)
    point_down = np.array(face[2][1].center)
    point_center = np.array(face[1][1].center)
    point_right = np.array(face[1][0].center)
    point_left = np.array(face[1][2].center)
    vec_center_up = point_up - point_center
    vec_center_right = point_right - point_center
    vec_center_down = point_down - point_center
    vec_center_left = point_left - point_center
    cos_angle_up_right = cos_angle(vec_center_up, vec_center_right)
    cos_angle_up_left = cos_angle(vec_center_up, vec_center_left)
    cos_angle_down_right = cos_angle(vec_center_down, vec_center_right)
    cos_angle_down_left = cos_angle(vec_center_down, vec_center_left)
    cos_angles = [cos_angle_up_left, cos_angle_up_right, cos_angle_down_left, cos_angle_down_right]
    ANGLE_BOUNDS = (-0.4, 0.4)
    for idx, cos_angle in enumerate(cos_angles):
        # TODO: ALSO MAKE return false
        # assert cos_angle > ANGLE_BOUNDS[0] and cos_angle < ANGLE_BOUNDS[1], f"The angle {idx} is not between 66 and 113 degrees, angle: {math.degrees(math.acos(cos_angle))}"
        if not (cos_angle > ANGLE_BOUNDS[0] and cos_angle < ANGLE_BOUNDS[1]):
            return False
    # TODO: Add other checks for cube face
    return True

def detect_face(frame, contours: List[np.ndarray], orientation_correction: bool = True) -> Face | None:
    """Find the face of the cube in the contours list."""

    assert len(frame.shape) == 3, "The image must be in BGR format, "
    if len(contours) < 9: # We need at least 9 contours to form a cube for now
        return None
    camera_matrix = orientation.build_camera_matrix((vp.ORIENTATION_ESTIMATED_FOV_W, vp.ORIENTATION_ESTIMATED_FOV_H), vp.WIDTH, vp.HEIGHT)
    contours_data = PreProcessingData(contours)
    # Check each contour and find 8 neighbors
    for i in range(len(contours)):
        if contours_data.orientation[i] is None:
            continue
        face = [contours[i]]
        ids = [i]
        (rotation_vector, translation_vector, normal_vector) = contours_data.orientation[i]
        relative_positions = [(features.contour_to_world_coordinates(contours_data.centers_of_mass[i], contours_data.centers_of_mass[i], camera_matrix, rotation_vector, translation_vector))]
        centers = [contours_data.centers_of_mass[i]]
        #Find contour orientation
        for j in range(len(contours)):
            if i == j or contours_data.orientation[j] is None:
                continue
            # Find the distance between the centers of mass
            distance = np.sqrt((contours_data.centers_of_mass[i][0] - contours_data.centers_of_mass[j][0])**2 + (contours_data.centers_of_mass[i][1] - contours_data.centers_of_mass[j][1])**2)
            # If the distance is less than 1/3 of the perimeter of the contours, then they are neighbors
            if distance < (cv.arcLength(contours[i], True)/3 + cv.arcLength(contours[j], True)/3) and features.contorus_area_similarity(contours[i], contours[j], 0.15):
                # Find position of the neighbor contour relative to the center contour
                face.append(contours[j])
                ids.append(j)
                relative_positions.append(features.contour_to_world_coordinates(contours_data.centers_of_mass[j], contours_data.centers_of_mass[i], camera_matrix, rotation_vector, translation_vector))
                centers.append(contours_data.centers_of_mass[j])
        if len(face) == 9:
            columns = assemble_face_data(frame, contours, contours_data, ids, relative_positions, orientation_correction)
            if columns is None:
                continue
            if not check_face_integrity(columns, i):
                continue
            # Could also skip
            return columns
        pass
    return None

def correct_face_orientation(arrangement: list[list[Face]], squares: list[Face]) -> list[list[Face]] | None:
    """Rotate the face so that the top row in the image is the top row of the face."""
    #TODO: This only takes into account the x-axis, it should also take into account the y-axis by saving the two best rotations and then comparing them on the y-axis
    # Reconstruct the face as it appears in the image
    image_orientation = __reorder_face(squares, lambda k: k.center)

    # Rotate until they have the same first column
    for i in range(4):
        test_columns = np.rot90(arrangement, i)
        if test_columns[0][0].id != image_orientation[0][0].id:
            continue
        if test_columns[0][1].id != image_orientation[0][1].id:
            continue
        if test_columns[0][2].id != image_orientation[0][2].id:
            continue
        return test_columns
    return None

def __reorder_face(face: list[FaceSquare], accesor: Callable[[FaceSquare], tuple[Number, Number]]) -> list[list[FaceSquare]]:
    squares = sorted(face, key=lambda k: accesor(k)[0])
    columns: list[list[FaceSquare]] = [sorted(squares[x:x+3],key= lambda k: accesor(k)[1]) for x in range(0, len(squares), 3)]
    return columns