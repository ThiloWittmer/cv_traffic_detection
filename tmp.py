import numpy as np
import cv2 as cv
MatLike = np.ndarray 

ANGLE_THRESHHOLD = 20

def junction_in_sight(img: MatLike, visualize:bool=False) -> bool:
    """
    Detects if a junction (T or X) is visible in the image using classic CV.
    Returns True if a junction is likely present, False otherwise.
    """
    # 1. Fixed ROI (lower 40% of the image)
    h = img.shape[0]
    roi = img

    roi_hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    lower_road = np.array([int(190/2), int(8*2.55), int(35*2.55)])   
    upper_road = np.array([int(200/2), int(14*2.55), int(45*2.55)])
    road_mask = cv.inRange(roi_hsv, lower_road, upper_road)
    road_ratio = np.sum(road_mask > 0) / road_mask.size
    if road_ratio < 0.2:
        return False
    # 2. Preprocessing
    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
    blur = cv.GaussianBlur(gray, (5, 5), 0)

    # 3. Edge Detection
    edges = cv.Canny(blur, 50, 150, apertureSize=3)

    # 4. Hough Line Detection
    lines = cv.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=30)
    if visualize:
        cv.imshow("Road mask", road_mask)
        vis = roi.copy()
        if len(vis.shape) == 2:  # grayscale to BGR for color lines
            vis = cv.cvtColor(vis, cv.COLOR_GRAY2BGR)
        if lines is not None:
            for x1, y1, x2, y2 in lines.reshape(-1, 4):
                cv.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv.imshow("ROI with detected lines", vis)
        cv.waitKey(0)
        cv.destroyAllWindows()
    if lines is None:
        return False

    # 5. Line Angle Analysis
    angles = []
    for x1,y1,x2,y2 in lines.reshape(-1,4):
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        angles.append(angle)

    # Cluster angles into bins: left, right, vertical
    left = [a for a in angles if a < -ANGLE_THRESHHOLD]
    right = [a for a in angles if a > ANGLE_THRESHHOLD]
    vertical = [a for a in angles if -ANGLE_THRESHHOLD<= a <= ANGLE_THRESHHOLD]

    # 6. Heuristic: Junction if at least 3 angle clusters have enough lines
    clusters = [len(left) > 2, len(right) > 2, len(vertical) > 2]
    if (len(left) > 3 and len(right) > 3):
        return True

    return False
