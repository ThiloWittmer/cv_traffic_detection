import cv2 as cv
import numpy as np
from enum import Enum
from pathlib import Path
MatLike = np.ndarray 
BoundingBox = tuple[int, int, int, int]
from sign import Sign
from matplotlib import pyplot as plt

templates = [str(f) for f in Path('Project_images/templates').iterdir() if f.is_file() and f.suffix == '.png']

sift = cv.SIFT.create()
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)

TEMPLATE_TO_SIGN = {
    "stop": Sign.STOP,
    "schild_zone_30": Sign.ZONE_30,
    "schild_zone_30_ende": Sign.ZONE_30_ENDE,
    "vorfahrt": Sign.VORF,
    "vorfahrt_gew": Sign.VORF_GEW,
    "vorfahrt_von_oben_nach_links": Sign.VORF_OBEN_LINKS,
    "vorfahrt_von_oben_nach_rechts": Sign.VORF_OBEN_RECHTS,
    "vorfahrt_von_unten_nach_links": Sign.VORF_UNTEN_LINKS,
    "vorfahrt_von_unten_nach_rechts": Sign.VORF_UNTEN_RECHTS,
    # Add more as needed
}

ZONE_RED_LOW = np.array([int(350/2), int(62*2.55), int(50*2.55)])
ZONE_RED_HIGH = np.array([int(360/2), int(95*2.55), int(80*2.55)])

# Precompute template SIFT features
CACHED_TEMPLATES = []
for template_path in templates:
    template_name = Path(template_path).stem.lower()
    sign_enum = TEMPLATE_TO_SIGN.get(template_name)
    if sign_enum is None:
        continue
    template_img = cv.imread(template_path, cv.IMREAD_GRAYSCALE)
    if template_img is None:
        continue
    kp_temp, des_temp = sift.detectAndCompute(template_img, np.ones_like(template_img, dtype=np.uint8))
    if des_temp is None:
        continue
    CACHED_TEMPLATES.append((sign_enum, template_img, kp_temp, des_temp))

def match_templates(img: MatLike) -> list[tuple[Sign, BoundingBox]]:
    MIN_MATCH_COUNT = 8
    matches_found = []
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    kp_img, des_img = sift.detectAndCompute(img_gray, np.ones_like(img_gray, dtype=np.uint8))
    if des_img is None:
        return matches_found

    for sign_enum, template_img, kp_temp, des_temp in CACHED_TEMPLATES:
        
        matches = bf.knnMatch(des_temp, des_img, k=2)
        good = []
        mmmm = [m[0] for m in matches]
        img_matches = cv.drawMatches(
            template_img, kp_temp,
            img_gray, kp_img,
            mmmm[:15], img.copy(),
            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        cv.imshow("Matches", img_matches)
        cv.waitKey(0)
        cv.destroyAllWindows()
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)
    
        if len(good) >= MIN_MATCH_COUNT:
            src_pts = np.array([kp_temp[m.queryIdx].pt for m in good], dtype=np.float32).reshape(-1, 1, 2)
            dst_pts = np.array([kp_img[m.trainIdx].pt for m in good], dtype=np.float32).reshape(-1, 1, 2)
            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
            if M is not None and mask is not None:
                inliers = mask.ravel().sum()
                if inliers / len(mask) < 0.6:
                    continue

                # img_matches = cv.drawMatches(
                #     template_img, kp_temp,
                #     img_gray, kp_img,
                #     good, img.copy(),
                #     matchesMask=mask.ravel().astype(int).tolist(),
                #     flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                # )
                # cv.imshow("Matches", img_matches)
                # cv.waitKey(0)
                # cv.destroyAllWindows()

                h, w = template_img.shape
                pts = np.array([[0.0, 0.0], [float(w), 0.0], [float(w), float(h)], [0.0, float(h)]], dtype=np.float32).reshape(-1, 1, 2)
                dst = cv.perspectiveTransform(pts, M)
                xs = dst[:, 0, 0]
                ys = dst[:, 0, 1]
                x, y, w_box, h_box = int(xs.min()), int(ys.min()), int(xs.max() - xs.min()), int(ys.max() - ys.min())
                matches_found.append((sign_enum, (x, y, w_box, h_box)))


    return matches_found

def sign_verification(img:MatLike, sign:Sign) -> bool:
    """please just pass in the cropped sign"""

    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    if sign == Sign.ZONE_30_ENDE:
        mask = cv.inRange(img_hsv, ZONE_RED_LOW, ZONE_RED_HIGH)
        flat = mask.flatten().astype(np.float32)
        avg = np.average(flat)
        return bool(avg < 10)
    return False
