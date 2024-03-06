from enum import Enum

from . import metafeatures
import cv2 as cv
import numpy as np

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

    def debug_image(self, dimensions: tuple[int, int] = (800, 600)):
        """Returns an image with the debug information of the state of the cube."""
        # Createa a black image
        img = np.zeros((dimensions[1], dimensions[0], 3), np.uint8)
        # Draw the faces


        return img
    def reset (self):
        self.faces = []
