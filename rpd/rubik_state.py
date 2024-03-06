from enum import Enum

from . import metafeatures
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

class SquareColor(str, Enum):
    """Enum to represent the color of a square in a face of the cube."""
    Unknown = 0
    WHITE = 1
    YELLOW = 2
    BLUE = 3
    GREEN = 4
    RED = 5
    ORANGE = 6

class RubikStateEngine:
    """Represents the state of a Rubik's cube.

    It can be fed with the faces from the detection engine and it will keep track of the state of the cube.
    """

    def __init__(self):
        self.faces = []
        pass

    def consume_face(self, face: metafeatures.Face):
        """Consumes a face from the detection engine and updates the state of the cube."""
        if face is None:
            raise ValueError("The face is None")
        if len(self.faces) == 6:
            raise ValueError("The cube is already complete")
        self.faces.append(
            {
                "rotation": 0,
                "color": SquareColor.Unknown, # Means the color of the center square
                "data": face
            }
        )
    def is_complete(self):
        """Returns True if the cube is complete, False otherwise."""
        return len(self.faces) == 6

    def fit(self):
        """Identifies the colors of the faces and rotates the faces to make a valid cube."""
        if not self.is_complete():
            raise ValueError("The cube is not complete")
        all_squares_avg_lab = []
        center_squares_avg_lab = []
        for face in self.faces:
            for i, square_row in enumerate(face["data"]):
                for j, square in enumerate(square_row):
                    all_squares_avg_lab.append(square.avg_lab)
                    if i == 1 and j == 1:
                        center_squares_avg_lab.append(square.avg_lab)
        all_squares_avg_lab = np.array(all_squares_avg_lab)
        all_squares_avg_lab = all_squares_avg_lab[:, 1:]
        center_squares_avg_lab = np.array(center_squares_avg_lab)
        center_squares_avg_lab = center_squares_avg_lab[:, 1:]
        assert all_squares_avg_lab.shape[0] == 54, f"something went wrong finding the squares, got {all_squares_avg_lab.shape[0]} expected 54"
        assert center_squares_avg_lab.shape[0] == 6, f"something went wrong finding the center squares, got {center_squares_avg_lab.shape[0]} expected 6"
        all_squares_avg_lab = np.float32(all_squares_avg_lab)
        center_squares_avg_lab = np.float32(center_squares_avg_lab)
        # Apply k-means to the to k-means to the faces to identify the colors, using the center square of each face as seed point
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv.kmeans(all_squares_avg_lab, 6, None, criteria, 1, cv.KMEANS_RANDOM_CENTERS)
        plt.scatter(all_squares_avg_lab[:, 0], all_squares_avg_lab[:, 1], c=labels, s=50, cmap='viridis')
        plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
        plt.scatter(center_squares_avg_lab[:, 0], center_squares_avg_lab[:, 1], c='red', s=100, alpha=0.5)
        for x, face in enumerate(self.faces):
            face["labels"] = []
            for i, square_row in enumerate(face["data"].faces):
                face["labels"].append([])
                for j, square in enumerate(square_row):
                    index = len(self.faces) * x + len(face["data"].faces) * i + j
                    face["labels"][i].append(labels[index][0])



    def debug_image(self, dimensions: tuple[int, int] = (800, 600)):
        """Returns an image with the debug information of the state of the cube."""
        # Createa a black image
        img = np.zeros((dimensions[1], dimensions[0], 3), np.uint8)
        # Draw the faces
        for idx, face in enumerate(self.faces):
            for i, row in enumerate(face["data"]):
                for j, square in enumerate(row):
                    color_met = square.avg_lab
                    color = cv.cvtColor(np.array([[color_met]], dtype=np.uint8),cv.COLOR_LAB2BGR)[0][0]
                    color = (float(color[0]), float(color[1]), float(color[2]))
                    rectangle_pos = (idx * 100 + i * 25, j * 25)
                    img = cv.rectangle(img, rectangle_pos, (rectangle_pos[0] + 25, rectangle_pos[1] + 25), color, -1)
                    img = cv.putText(img, f"{face['labels'][i][j]}", (rectangle_pos[0] + 5, rectangle_pos[1] + 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)

        return img

    def reset (self):
        self.faces = []
