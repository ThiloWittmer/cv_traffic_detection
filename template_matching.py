import cv2 as cv
import numpy as np
from enum import Enum
from pathlib import Path
MatLike = np.ndarray 

BoundingBox = tuple[int, int, int, int]
"""[x,y,w,h]"""

from enums import Sign
from matplotlib import pyplot as plt

templates = [str(f) for f in Path('Project_images/templates').iterdir() if f.is_file() and f.suffix == '.png']

sift = cv.SIFT.create(nfeatures=2000)
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
}

ZONE_RED_LOW = np.array([int(350/2), int(62*2.55), int(50*2.55)])
ZONE_RED_HIGH = np.array([int(360/2), int(95*2.55), int(80*2.55)])

VORF_YEL_LOW =np.array([int(40/2), int(60*2.55), int(60*2.55)])
VORF_YEL_HIGH =np.array([int(70/2), int(105*2.55), int(105*2.55)])

# Precompute template SIFT features
CACHED_TEMPLATES = []
CACHED_TEMPLATES_WITH_KP = []
for template_path in templates:
    template_name = Path(template_path).stem.lower()
    sign_enum = TEMPLATE_TO_SIGN.get(template_name)
    if sign_enum is None:
        continue
    template_img = cv.imread(template_path)
    template_img_gray = cv.cvtColor(template_img, cv.COLOR_BGR2GRAY)
    if template_img is None:
        continue
    CACHED_TEMPLATES.append((template_img, sign_enum))
    kp_temp, des_temp = sift.detectAndCompute(template_img_gray, np.ones_like(template_img_gray, dtype=np.uint8))
    if des_temp is None:
        continue
    CACHED_TEMPLATES_WITH_KP.append((sign_enum, template_img_gray, kp_temp, des_temp))

def match_templates(img: MatLike) -> list[tuple[Sign, BoundingBox]]:
    results_sift =  match_templates_with_sift(img)
    results_shape = match_templates_with_shape(img)

    return results_sift + results_shape

def match_templates_with_shape(img: MatLike) -> list[tuple[Sign, BoundingBox]]:
    matches_found = []
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    img_blur = cv.GaussianBlur(img_gray, (5, 5), 0)
    _, img_bin = cv.threshold(img_blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    contours_img, _ = cv.findContours(img_bin, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for template_bgr, sign_enum in CACHED_TEMPLATES:
        template_gray = cv.cvtColor(template_bgr, cv.COLOR_BGR2GRAY) if len(template_bgr.shape) == 3 else template_bgr
        template_blur = cv.GaussianBlur(template_gray, (5, 5), 0)
        _, template_bin = cv.threshold(template_blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        contours_template, _ = cv.findContours(template_bin, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not contours_template:
            continue
        template_cnt = max(contours_template, key=cv.contourArea)

        best_score = float('inf')
        best_box = None

        for cnt in contours_img:
            if cv.contourArea(cnt) < 500:  # filter out small contours
                continue
            score = cv.matchShapes(template_cnt, cnt, cv.CONTOURS_MATCH_I1, 0.0)
            if score < 0.2:  # threshold for a good match, adjust as needed
                x, y, w, h = cv.boundingRect(cnt)
                if score < best_score:
                    best_score = score
                    best_box = (x, y, w, h)

        if best_box is not None:
            matches_found.append((sign_enum, best_box))

    return matches_found

def match_templates_with_sift(img: MatLike) -> list[tuple[Sign, BoundingBox]]:
    MIN_MATCH_COUNT = 4
    matches_found = []
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    kp_img, des_img = sift.detectAndCompute(img_gray, np.ones_like(img_gray, dtype=np.uint8))
    if des_img is None:
        return matches_found
    
        

    for sign_enum, template_img, kp_temp, des_temp in CACHED_TEMPLATES_WITH_KP:

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
                # cv.waitKey()
                # cv.destroyAllWindows()

                h, w = template_img.shape
                pts = np.array([[0.0, 0.0], [float(w), 0.0], [float(w), float(h)], [0.0, float(h)]], dtype=np.float32).reshape(-1, 1, 2)
                dst = cv.perspectiveTransform(pts, M)
                xs = dst[:, 0, 0]
                ys = dst[:, 0, 1]
                x, y, w_box, h_box = int(xs.min()), int(ys.min()), int(xs.max() - xs.min()), int(ys.max() - ys.min())
                if x < 0 or y < 0 or w_box > img.shape[1] or h_box > img.shape[0] or w_box < 10 or h_box < 10:
                    continue
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
    
    if sign == Sign.VORF:
        mask = cv.inRange(img_hsv, VORF_YEL_LOW, VORF_YEL_HIGH)
        flat = mask.flatten().astype(np.float32)
        avg = np.average(flat)
        return bool(avg > 10)

    return False
