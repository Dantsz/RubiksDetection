import numpy as np
import logging

class SolutionDisplayEngine:

    def __init__(self):
        self.reset()

    def reset(self):
        self.centers = None
        self.solving_moves = None
        self.frame = 0

    def consume_solution(self, detected_centers, solving_moves: list[str]):
        self.centers = detected_centers
        self.solving_moves = solving_moves

    def ready(self):
        return self.centers is not None and self.solving_moves is not None

    def display_solution(self, frame: np.ndarray, face: list[int]) -> np.ndarray:

        logging.info(f"Solution display: displaying solution, frame {self.frame}")
        self.frame += 1
        return frame
        #Flow:
        #face -> colors
        #based on solution, display move