import numpy as np
import cv2 as cv
MatLike = np.ndarray 

def junction_in_sight(img: MatLike, visualize:bool=False) -> bool:
    """
    Detects if a junction (T or X) is visible in the image using the shape of the road mask.
    Returns True if a junction is likely present, False otherwise.
    """
    # 1. Fixed ROI (lower 40% of the image)
    h = img.shape[0]
    roi = img[int(h*0.6):, :]

    # 2. Road color mask (HSV)
    roi_hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    lower_road = np.array([int(190/2), int(8*2.55), int(35*2.55)])   
    upper_road = np.array([int(200/2), int(14*2.55), int(45*2.55)])
    road_mask = cv.inRange(roi_hsv, lower_road, upper_road)

    # 3. Morphological cleaning
    kernel = np.ones((7, 7), np.uint8)
    road_mask_clean = cv.morphologyEx(road_mask, cv.MORPH_CLOSE, kernel)

    # 4. Find contours
    contours, _ = cv.findContours(road_mask_clean, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        if visualize:
            cv.imshow("Road mask", road_mask_clean)
            cv.waitKey(0)
            cv.destroyAllWindows()
        return False

    # 5. Find the largest contour (main road)
    largest_contour = max(contours, key=cv.contourArea)
    mask_main_road = np.zeros_like(road_mask_clean)
    cv.drawContours(mask_main_road, [largest_contour], -1, (255,), thickness=cv.FILLED)

    # 6. Analyze splits at the top and middle of the mask
    h_mask, w_mask = mask_main_road.shape
    split_rows = [int(h_mask*0.1), int(h_mask*0.5)] 
    split_counts = []
    for row in split_rows:
        line = mask_main_road[row:row+10, :]
        n_labels, _ = cv.connectedComponents(line)
        split_counts.append(n_labels - 1)  

    is_junction = any(count >= 2 for count in split_counts)

    # 7. Visualization
    if visualize:
        vis = cv.cvtColor(mask_main_road, cv.COLOR_GRAY2BGR)
        cv.drawContours(vis, [largest_contour], -1, (0,255,0), 2)
        for row in split_rows:
            cv.line(vis, (0, row), (w_mask-1, row), (255,0,0), 1)
        cv.imshow("Road mask with splits", vis)
        cv.waitKey(0)
        cv.destroyAllWindows()

    return is_junction
