import cv2 as cv
import numpy as np

def extract_lines_houghP(img: cv.Mat) -> [cv.typing.MatLike]:
    lines = cv.HoughLinesP(img, 1, np.pi/45, threshold=50, minLineLength=100, maxLineGap=5)
    if lines is None:
        return []
    lines = [line[0] for line in lines]
    return lines

# Return list of pair of lines that are perpendicular
def filter_perpendicular_lines(lines: np.ndarray) -> [(np.ndarray,np.ndarray)]:
    # Filter lines by slope
    filtered_lines = []
    # iterate over pairs of lines
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            # Get the coordinates of the lines
            x1, y1, x2, y2 = lines[i]
            x3, y3, x4, y4 = lines[j]
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
                filtered_lines.append((lines[i],lines[j]))
    return filtered_lines

def find_intersection_points(lines:[(np.ndarray,np.ndarray)]) -> [np.ndarray]:
    'FInds the intersection points of the two lines for each entry in the array, returns None if no intersection, the index of the entry matches the index of the intersection point'
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

def point_merge(points : [np.ndarray], lines : [(np.ndarray, np.ndarray)], distance: float) -> [np.ndarray]:
    assert len(points) == len(lines)
    for i in range(len(points)):
        if points[i] is None:
            continue
        for j in range(i+1, len(points)):
            if points[j] is None:
                continue
            if np.linalg.norm(points[i]-points[j]) < distance:
                points[j] = points[i]
    return points

def for_each_line_pair(lines: [(np.ndarray,np.ndarray)], func: callable) -> [np.ndarray]:
    'Applies func to each pair of lines in lines, returns the result of func'
    results = []
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            results.append(func(lines[i], lines[j]))
    return results

def line_endpoints_distance(line1: np.ndarray, line2: np.ndarray) -> np.ndarray:
    'Returns the distance between the endpoints of the two lines'
    distances = np.zeros((2, 2))
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    distances[0, 0] = np.linalg.norm(np.array((x1, y1)) - np.array((x3, y3)))
    distances[0, 1] = np.linalg.norm(np.array((x1, y1)) - np.array((x4, y4)))
    distances[1, 0] = np.linalg.norm(np.array((x2, y2)) - np.array((x3, y3)))
    distances[1, 1] = np.linalg.norm(np.array((x2, y2)) - np.array((x4, y4)))
    return distances

def line_proximity(line1: np.ndarray, line2: np.ndarray, threshold: float) -> bool:
    'Returns true if the endpoints of the two lines are within threshold distance of each other'
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    return np.linalg.norm(np.array((x1, y1)) - np.array((x3, y3))) < threshold or np.linalg.norm(np.array((x1, y1)) - np.array((x4, y4))) < threshold or np.linalg.norm(np.array((x2, y2)) - np.array((x3, y3))) < threshold or np.linalg.norm(np.array((x2, y2)) - np.array((x4, y4))) < threshold

def contours_filter_small_area(contours: [np.ndarray], threshold: float) -> [np.ndarray]:
    'Returns the contours that are larger than threshold'
    return [contour for contour in contours if cv.contourArea(contour) > threshold]

def contours_filter_large(contours: [np.ndarray], threshold: float) -> [np.ndarray]:
    'Returns the contours that are smaller than threshold'
    return [contour for contour in contours if cv.contourArea(contour) < threshold]

def contours_filter_convex(contours: [np.ndarray]) -> [np.ndarray]:
    'Returns the contours that are convex'
    return [contour for contour in contours if cv.isContourConvex(contour)]

def contours_filter_solidity(contours: [np.ndarray], threshold: float) -> [np.ndarray]:
    'Returns the contours that have solidity larger than threshold'
    return [contour for contour in contours if cv.contourArea(contour)/cv.contourArea(cv.convexHull(contour)) > threshold]

def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def contours_filter_positional_2(contours: [np.ndarray], threshold: float) -> [np.ndarray]:
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
            if distance(center1, center2) < threshold*(cv.arcLength(contour1, True)/3):
                filtered_contours.append(contour1)
                filtered_contours.append(contour2)

    return filtered_contours
