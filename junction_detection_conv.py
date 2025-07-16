import numpy as np
import cv2 as cv
from skimage.util import img_as_bool
from skimage.morphology import skeletonize, remove_small_objects
MatLike = np.ndarray 
from matplotlib import pyplot as plt
from scipy.ndimage import convolve

def junction_in_sight(img: MatLike, visualize:bool=False) -> bool:
    """
    Detects if a junction (T or X) is visible in the image using the shape of the road mask.
    Returns True if a junction is likely present, False otherwise.
    """
    h = img.shape[0]
    roi = img[int(h*0.2):, :]

    roi_hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    lower_road = np.array([int(190/2), int(8*2.55), int(35*2.55)])   
    upper_road = np.array([int(200/2), int(14*2.55), int(45*2.55)])
    road_mask = cv.inRange(roi_hsv, lower_road, upper_road)

    kernel_close = np.ones((20, 20), np.uint8)
    road_mask_clean = cv.morphologyEx(road_mask, cv.MORPH_CLOSE, kernel_close, iterations=5)

    C = 20
    road_mask_bool = img_as_bool(road_mask_clean)
    skel = skeletonize(road_mask_bool)
    skel_clean = remove_small_objects(skel, min_size=100)
    kernel = np.array([[1,1,1],[1,C,1],[1,1,1]])
    filtered = convolve(skel_clean.astype(np.uint8), kernel, mode='constant')
    endpoints = np.sum(filtered==C+1)
    branchpoints = np.sum(filtered >= C+3)

    if (endpoints > 2):
        return True  
    
    # plt.subplot(1,2,1)
    # plt.imshow(roi, cmap='gray')
    # plt.title('skel')
    # plt.xticks([])
    # plt.yticks([])
    # plt.subplot(1,2,2)
    # plt.imshow(skel, cmap='gray')
    # plt.title('skel')
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()
    return False