import cv2 as cv
import numpy as np

def extract_lines_houghP(img: cv.Mat) -> [cv.typing.MatLike]:
    lines = cv.HoughLinesP(img, 0.5, np.pi/180, threshold=50, minLineLength=100, maxLineGap=5)
    return lines if lines is not None else []

# Return list of pair of lines that are perpendicular
def filter_perpendicular_lines(lines: np.ndarray) -> [(np.ndarray,np.ndarray)]:
    # Filter lines by slope
    filtered_lines = []
    # iterate over pairs of lines
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            # Get the coordinates of the lines
            x1, y1, x2, y2 = lines[i][0]
            x3, y3, x4, y4 = lines[j][0]
            #Calculate dot product of the two lines
            dot_product = (x2-x1)*(x4-x3)+(y2-y1)*(y4-y3)
            #Calculate the length of the lines
            length1 = np.sqrt((x2-x1)**2+(y2-y1)**2)
            length2 = np.sqrt((x4-x3)**2+(y4-y3)**2)
            #Check lengths are not zero
            if length1 == 0 or length2 == 0:
                continue
            cos = dot_product/(length1*length2)
            if cos > 1:
                cos = 1
            elif cos < -1:
                cos = -1
            #Calculate the angle between the lines
            angle = np.arccos(cos)
            #Convert to degrees
            angle = np.degrees(angle)
            if angle in range(80, 100):
                filtered_lines.append((lines[i][0],lines[j][0]))
    return filtered_lines

def find_intersection_points(lines) -> [np.ndarray]:
    intersection_points = []
    for i in range(len(lines)):
            (p1, p2, p3, p4), (q1, q2, q3, q4) = lines[i]
            A = np.array([
                [p3 - p1, -(q3 - q1)],
                [p4 - p2, -(q4 - q2)]
            ])
            B = np.array([q1 - p1, q2 - p2])
            # Solve the linear system
            try:
                t = np.linalg.solve(A, B)
            except np.linalg.LinAlgError:
                continue  # Lines are parallel, no intersection

            # Check if the intersection point is within the line segments
            if 0 <= t[0] <= 1 and 0 <= t[1] <= 1:
                point1 = np.array((p1,p2))
                point2 = np.array((p3,p4))
                intersection_point = point1 + t[0] * (point2 - point1)
                intersection_points.append(intersection_point)
            else:
                intersection_points.append(None)
    return intersection_points

def point_merge(points : [(np.ndarray,np.ndarray)],lines : np.ndarray, distance) -> [np.ndarray]:
    pass