"""Feature extraction functions for Rubik's cube detection, including contour filtering, polygon approximation, and perspective transformation."""
import cv2 as cv
import numpy as np
from . import viewport_properties as vp
from typing import List, Tuple

def contours_filter_small_area(contours: List[np.ndarray], threshold: float) -> List[np.ndarray]:
    'Returns the contours that are larger than threshold'
    return [contour for contour in contours if cv.contourArea(contour) > threshold]

def contours_filter_large(contours: List[np.ndarray], threshold: float) -> List[np.ndarray]:
    'Returns the contours that are smaller than threshold'
    return [contour for contour in contours if cv.contourArea(contour) < threshold]

def contours_filter_convex(contours: List[np.ndarray]) -> List[np.ndarray]:
    'Returns the contours that are convex'
    return [contour for contour in contours if cv.isContourConvex(contour)]

def contours_filter_solidity(contours: List[np.ndarray], threshold: float) -> List[np.ndarray]:
    'Returns the contours that have solidity larger than threshold'
    return [contour for contour in contours if cv.contourArea(contour)/cv.contourArea(cv.convexHull(contour)) > threshold]

def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def contours_filter_isolated_contours(contours: List[np.ndarray], threshold: float) -> List[np.ndarray]:
    'Returns the contours that are within (perimeter/4) distance from the center of mass of a another contour'
    filtered_contours = []
    for i in range(len(contours)):
        for j in range(i+1, len(contours)):
            contour1 = contours[i]
            contour2 = contours[j]
            M1 = cv.moments(contour1)
            M2 = cv.moments(contour2)
            center1 = (int(M1["m10"] / M1["m00"]), int(M1["m01"] / M1["m00"]))
            center2 = (int(M2["m10"] / M2["m00"]), int(M2["m01"] / M2["m00"]))
            if distance(center1, center2) < threshold*(cv.arcLength(contour1, True)/4 + cv.arcLength(contour2, True)/4):
                if not any(np.array_equal(contour1, contour) for contour in filtered_contours):
                    filtered_contours.append(contour1)
                if not any(np.array_equal(contour2, contour) for contour in filtered_contours):
                    filtered_contours.append(contour2)

    return filtered_contours

def approx_polygon_from_contour(contours: List[np.ndarray] , epsilon: float = vp.FEATURES_POLY_APPROX_DEFAULT_EPSILON) -> np.ndarray:
    'Returns the approximated polygon of the contours'
    return [cv.approxPolyDP(contour, epsilon, True) for contour in contours]

def contours_filter_vertices(contours: List[np.ndarray], threshold: int = 2) -> List[np.ndarray]:
    'Returns the contours that have number of vertices larger than threshold'
    return [contour for contour in contours if len(contour) > threshold]

def contours_min_area_rect(contours: List[np.ndarray]) -> List[np.ndarray]:
    'Returns the minimum area rectangle of the contours'
    return [np.int0(cv.boxPoints(cv.minAreaRect(contour))) for contour in contours]

def contorus_area_similarity(contour1: np.ndarray, contour2: np.ndarray, threshold: float) -> bool:
    'Returns True if the contours are similar, '
    area_ratio = min(cv.contourArea(contour1), cv.contourArea(contour2))/max(cv.contourArea(contour1), cv.contourArea(contour2))
    return (1 - area_ratio) < threshold

def contours_crop_and_reverse_perspective(image, contours: List[np.ndarray], image_size : Tuple[int,int]) -> List[np.ndarray]:
    '''Returns the cropped parts of the image that are inside the bounding boxes of the contours'''
    cropped_images = []
    image_width, image_height = image_size
    for contour in contours:
        box = np.int0(cv.boxPoints(cv.minAreaRect(contour)))
        pts1 = np.float32(box)
        pts2 = np.float32([[0, 0], [image_width, 0], [image_width, image_height], [0, image_height]])
        matrix = cv.getPerspectiveTransform(pts1, pts2)
        cropped_images.append(cv.warpPerspective(image, matrix, (image_width, image_height)))
    return cropped_images

def contour_to_world_coordinates(contour_center: Tuple[int,int], origin : Tuple[int,int], camera_matrix: np.ndarray, rotation_vector: np.ndarray, translation_vector: np.ndarray) -> Tuple[float, float]:
    '''Returns the contour in the basis of the rotation and translation vectors'''
    obj_points = np.float32([[contour_center[0], contour_center[1], 1] ]).reshape(-1,3)
    rotation_matrix, _ = cv.Rodrigues(rotation_vector)
    inverse_rotation_matrix = rotation_matrix.transpose()
    # relative_point, _ = cv.projectPoints(obj_points, inverse_rotation_vector, -translation_vector, camera_matrix, None)
    inverse_camera = np.linalg.inv(camera_matrix)
    relative_point = inverse_camera @ obj_points.T
    relative_point =  inverse_rotation_matrix @  (relative_point - translation_vector)
    relative_point = relative_point.squeeze()
    relative_point = np.array([relative_point[0] / abs(relative_point[2])  , relative_point[1] - abs(relative_point[2])])
    assert relative_point.shape == (2,), "The relative point should have 2 dimensions"
    return relative_point