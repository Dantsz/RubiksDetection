from enum import Enum

from RubiksDetection.rpd import cube_state

from . import metafeatures

import logging
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
logging.getLogger("PIL").setLevel(logging.INFO)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from pydantic import BaseModel
from .color import SquareColor, reference_colors

def check_label_consistency(labels: np.ndarray) -> bool:
    """Checks if the labels make sense.

        Each label must appear exactly 9 times in the list.
    """
    for i in range(6):
        sm = np.sum(labels == i)
        if sm != 9:
            return False
    return True

def classify_squares_k_means(squares: list[metafeatures.FaceSquare], centers: list[tuple[float, float, float]]) -> tuple[np.ndarray, np.ndarray]:
    """Classifies the squares of the cube state using k-means clustering.

    The squares parameter is a list of average LAB values for each square in the cube, the centers parameter represents the square at index (1,1) of each face.
    The first return value is a list of integers representing a label for each square, in order that they're represented in the squares list.
    The second return value is a list of the centers of the clusters.
    """
    squares = np.float32(squares)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv.kmeans(squares, 6, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

    return labels, centers

def classify_squares_closest(squares: list[metafeatures.FaceSquare], centers: list[tuple[float, float, float]]) -> tuple[np.ndarray, np.ndarray]:
    """Classifies the squares of the cube state using the closest center to each square.

    The squares parameter is a list of average LAB values for each square in the cube, the centers parameter represents the square at index (1,1) of each face.
    The return value is a list of integers representing a label for each square, in order that they're represented in the squares list.
    """
    labels = []
    for square in squares:
        min_distance = float('inf')
        label = -1
        for i, center in enumerate(centers):
            distance = np.linalg.norm(np.array(center) - np.array(square))
            if distance < min_distance:
                min_distance = distance
                label = i
        labels.append([label])
    return np.array(labels), centers

def fit_colors_to_labels(labels: list[int], centers: np.ndarray) -> list[SquareColor]:
    """Fits the labels to the closest color in the reference colors.
    """
    colors: list[SquareColor] = []
    for center in centers[: , 1:]:
        min_distance = float('inf')
        color = SquareColor.Unknown
        for i, ref_color in enumerate(reference_colors) :
            distance = np.linalg.norm(np.array(ref_color) - np.array(center))
            if distance < min_distance and SquareColor(i) not in colors:
                min_distance = distance
                color = SquareColor(i)
        colors.append(color)
    return colors

class LabelingEngine:
    """Labels the state of the cube.

    It can be fed with the faces from the detection engine and it will keep track of the state of the cube.
    """

    def __init__(self):
        self.face_data: list[metafeatures.FaceSquare] = []
        self.face_labels: list[list[list[SquareColor]]] = []
        self.last_centers = []
        self.colors: list[SquareColor] = []
        self.center_labels: list[int] = []
        pass

    def consume_face(self, face: metafeatures.Face):
        """Consumes a face from the detection engine and updates the state of the cube."""
        if face is None:
            raise ValueError("The face is None")
        if len(self.face_data) == 6:
            raise ValueError("The cube is already complete")
        self.face_data.append(face)

    def is_complete(self):
        """Returns True if the cube is complete, False otherwise."""
        return len(self.face_data) == 6

    def fit(self):
        """Identifies the colors of the faces and rotates the faces to make a valid cube."""
        if not self.is_complete():
            raise ValueError("The cube is not complete")
        all_squares_avg_lab: list[tuple[float, float, float]] = []
        center_squares_avg_lab: list[tuple[float, float, float]] = []
        for face in self.face_data:
            for i, square_row in enumerate(face):
                for j, square in enumerate(square_row):
                    all_squares_avg_lab.append(square.avg_lab)
                    if i == 1 and j == 1:
                        center_squares_avg_lab.append(square.avg_lab)

        all_squares_avg_lab = np.array(all_squares_avg_lab)
        center_squares_avg_lab = np.array(center_squares_avg_lab)
        assert all_squares_avg_lab.shape[0] == 54, f"something went wrong finding the squares, got {all_squares_avg_lab.shape[0]} expected 54"
        assert center_squares_avg_lab.shape[0] == 6, f"something went wrong finding the center squares, got {center_squares_avg_lab.shape[0]} expected 6"

        labels, centers = classify_squares_closest(all_squares_avg_lab, center_squares_avg_lab)

        if check_label_consistency(labels):
            logging.info("The labels are consistent")
        else:
            logging.warning("The labels are not consistent")

        self.last_centers = centers
        self.colors = fit_colors_to_labels(labels, self.last_centers)
        self.color_centers = list(range(6))
        for i in range(6):
            self.color_centers[int(self.colors[i])] = self.last_centers[i]

        for x, face in enumerate(self.face_data):
            self.face_labels.append([])
            for i, square_row in enumerate(face.faces):
                self.face_labels[x].append([])
                for j, square in enumerate(square_row):
                    index = 9 * x + len(face.faces) * i + j
                    self.face_labels[x][i].append(self.colors[labels[index][0]])
                    if i == 1 and j == 1:
                        self.center_labels.append(self.colors[labels[index][0]])


    def state(self) -> cube_state.CubeState:
        """Returns the state of the cube as a CubeState object."""
        if not self.is_complete():
            raise ValueError("The cube is not complete")
        return cube_state.CubeState(np.array(self.face_labels))

    def stateString(self) -> str:
        if not self.is_complete():
            raise ValueError("The cube is not complete")
        return cube_state.CubeState(np.array(self.face_labels)).to_solver_string()

    def debug_image(self, dimensions: tuple[int, int] = (800, 100)):
        """Returns an image with the debug information of the state of the cube."""
        logging.info("Creating debug image")
        plt.clf()
        # Createa a black image
        img = np.zeros((dimensions[1], dimensions[0], 3), np.uint8)
        # Draw the faces
        avg_points = []
        colors = []
        for idx, face in enumerate(self.face_data):
            for i, row in enumerate(face):
                for j, square in enumerate(row):
                    color_met = square.avg_lab
                    color = cv.cvtColor(np.array([[color_met]], dtype=np.uint8),cv.COLOR_LAB2BGR)[0][0]
                    color = (float(color[0]), float(color[1]), float(color[2]))
                    rectangle_pos = (idx * 100 + i * 25, j * 25)
                    img = cv.rectangle(img, rectangle_pos, (rectangle_pos[0] + 25, rectangle_pos[1] + 25), color, -1)
                    img = cv.putText(img, f"{int(self.face_labels[idx][i][j])}", (rectangle_pos[0] + 5, rectangle_pos[1] + 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
                    avg_points.append(face[i][j].avg_lab)

                    color_rgb = cv.cvtColor(np.array([[color_met]], dtype=np.uint8),cv.COLOR_LAB2RGB)[0][0]
                    color_rgb = (float(color[0]), float(color[1]), float(color[2]))
                    colors.append(np.array(color_rgb, dtype=np.float16))

        x = [x[1] for x in avg_points]
        y = [x[2] for x in avg_points]
        normalized_colors = np.array(colors) / 255
        plt.scatter(x, y,c=normalized_colors)

        x = [x[1] for x in self.last_centers]
        y = [x[2] for x in self.last_centers]
        plt.scatter(x, y, c='red')

        for fi, face in enumerate(self.face_data):
            for i, square_row in enumerate(face):
                for j, square in enumerate(square_row):
                    if i == 1 and j == 1:
                        plt.annotate(self.face_labels[fi][i][j].name, (square.avg_lab[1], square.avg_lab[2]), textcoords="offset points", xytext=(0,10), ha='center')


        fig = plt.gcf()
        fig.canvas.draw()
        plot_img = np.array(fig.canvas.renderer.buffer_rgba())
        plot_img = cv.cvtColor(plot_img, cv.COLOR_RGBA2BGR)
        plot_img = cv.resize(plot_img, (800,600))

        img =  cv.vconcat([img, plot_img])

        return img

    def reset (self):
        self.face_data = []
        self.face_labels = []
        self.last_centers = []
