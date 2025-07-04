from map_detection import detect_turns
from pathlib import Path
from enum import Enum
import cv2 as cv
import numpy as np
from ultralytics import YOLO
from template_matching import match_templates
from junction_detection import junction_in_sight

#for type-hints
MatLike = np.ndarray 
BoundingBox = tuple[int, int, int, int]

yolo_model = YOLO("yolov8l.pt")

dir_imgs = Path('Project_images')
templates = [str(f) for f in Path('Project_images/templates').iterdir() if f.is_file() and f.suffix == '.png']
scenes_tmp = [[str(img) for img in scene.iterdir() if img.is_file() and img.suffix == '.png'] for scene in dir_imgs.iterdir() if scene.is_dir() and scene.name.startswith('Szene')]
stadtkarte = 'Project_images/stadtkarte/stadtplan_ohne_gebaeude.png'
scenes: dict[str,list[str]] = {}
for scene in scenes_tmp:
    k = scene[0].split('/')[1]
    scenes[k] = scene
current_speed_limit = 50
result = {}

class Sign(Enum):
    ZONE_30 =           1
    ZONE_30_ENDE =      2
    STOP =              3
    VORF =              4
    VORF_GEW =          5
    VORF_OBEN_LINKS =   6
    VORF_OBEN_RECHTS =  7
    VORF_UNTEN_LINKS =  8
    VORF_UNTEN_RECHTS = 9

class TrafficColor(Enum):
    RED =           1
    YELLOW =        2
    RED_YELLOW =    3
    GREEN =         4

RED_OFF_LOW = np.array([int(350/2), int(60*2.55), int(38*2.55)])
RED_OFF_HIGH= np.array([int(359/2), int(90*2.55), int(50*2.55)])
YELLOW_OFF_LOW = np.array([int(47/2), int(40*2.55), int(40*2.55)])
YELLOW_OFF_HIGH = np.array([int(53/2), int(60*2.55), int(50*2.55)])
GREEN_OFF_LOW = np.array([int(152/2), int(70*2.55), int(30*2.55)])
GREEN_OFF_HIGH = np.array([int(162/2), int(85*2.55), int(40*2.55)])

RED_ON_LOW = np.array([int(354/2), int(81*2.55), int(95*2.55)])
RED_ON_HIGH = np.array([int(360/2), int(90*2.55), int(105*2.55)])
YELLOW_ON_LOW = np.array([int(56/2), int(77*2.55), int(95*2.55)])
YELLOW_ON_HIGH = np.array([int(64/2), int(83*2.55), int(105*2.55)])
GREEN_ON_LOW = np.array([int(100/2), int(75*2.55), int(80*2.55)])
GREEN_ON_HIGH = np.array([int(130/2), int(105*2.55), int(105*2.55)])

def object_detection(img_paths: list[str]) -> list[list[tuple[str, BoundingBox]]]:
    """run this for whole scene\n
    returns detected objects for all images in scene\n
    returns [[(class_name, (x,y,w,h)), ...], [(class_name, (x,y,w,h)), ....]]"""
    results = yolo_model(img_paths)
    results_all_images = []
    for r in results:
        results_one_image = []
        boxes = r.boxes.xyxy.cpu().numpy()  
        classes = r.boxes.cls.int().cpu().numpy()
        for box, cls_id in zip(boxes, classes):
            class_name = r.names[int(cls_id)]
            x1, y1, x2, y2 = np.round(box).astype(int)
            x, y = x1, y1
            w, h = x2 - x1, y2 - y1
            bbox = (x, y, w, h)
            results_one_image.append((class_name, bbox))
        results_all_images.append(results_one_image)
    
    return results_all_images

def sign_verification(img:MatLike, bounding_box:BoundingBox):
    return

def colorPresent(mask: MatLike) -> bool:
    PRESENT_THRESHHOLD = 30
    flat = mask.flatten()
    return bool(np.average(flat) > PRESENT_THRESHHOLD)

def get_traffic_light_color(img: MatLike) -> TrafficColor | None:
    """please just pass in the cropped traffic light"""
    red_on: bool =False
    yellow_on: bool = False
    green_on:bool = False

    h,w, _ = img.shape
    h_third = int(h/3)
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    red_roi = img_hsv[0:h_third, :w]
    yellow_roi = img_hsv[h_third:h_third*2, :w]
    green_roi = img_hsv[h_third*2:h, :w]

    red_off_mask = cv.inRange(red_roi, RED_OFF_LOW, RED_OFF_HIGH)
    yellow_off_mask = cv.inRange(yellow_roi, YELLOW_OFF_LOW, YELLOW_OFF_HIGH)
    green_off_mask = cv.inRange(green_roi, GREEN_OFF_LOW, GREEN_OFF_HIGH)

   
    red_off_present = colorPresent(red_off_mask)
    yellow_off_present = colorPresent(yellow_off_mask)
    green_off_present = colorPresent(green_off_mask)

    if (not red_off_present and not yellow_off_present and not green_off_present):
        return None

    red_on_mask = cv.inRange(red_roi, RED_ON_LOW, RED_ON_HIGH)
    yellow_on_mask = cv.inRange(yellow_roi, YELLOW_ON_LOW, YELLOW_ON_HIGH)
    green_on_mask = cv.inRange(green_roi, GREEN_ON_LOW, GREEN_ON_HIGH)

    red_on = colorPresent(red_on_mask)
    yellow_on = colorPresent(yellow_on_mask)
    green_on = colorPresent(green_on_mask)

    if (green_on and yellow_off_present and red_off_present):
        return TrafficColor.GREEN
    
    if (red_on and green_off_present and yellow_off_present):
        return TrafficColor.RED
    
    if (yellow_on and green_off_present and red_off_present):
        return TrafficColor.YELLOW
    
    if (red_on and yellow_on and green_off_present):
        return TrafficColor.RED_YELLOW
    
    # cv.imshow("r", cv.cvtColor(red_roi, cv.COLOR_HSV2BGR))
    # cv.imshow("y", yellow_roi)
    # cv.imshow("g", green_roi)
    # cv.waitKey()
    # cv.destroyAllWindows()
    return None 

def get_image_crop(img:MatLike, box_cutout: BoundingBox) -> MatLike:
    x,y,w,h = box_cutout
    return img[y : y+h, x : x+w]

def process_scene(turn:str, img_paths:list[str]) -> list[dict]:

    return []


def main():
    # TEST template matching
    # imgs = scenes.get("Szene_4")
    # if img:
    #     img = img[4]
    #     test = "Project_images/templates/stop.png"
    #     img = cv.imread(img)
    #     cv.imshow("test", img)
    #     cv.waitKey()
    #     cv.destroyAllWindows()
    #     for match in match_templates(img):
    #         print(f"{match[0]}: {match[1]}")
    
    # TEST Kreuzung erkennen
    # for k, imgs in scenes.items():
    #     if imgs:
    #         for i, img in enumerate(imgs):
    #             img = cv.imread(img)    
    #             print(f"{i}: {junction_in_sight(img, visualize=True)}")


    # TEST ampel farbe erkennen
    scene = scenes.get("Szene_3")
    if scene:
        for img_path in scene:
            img = cv.imread(img_path)
            cv.imshow("test", img)
            detected_objects = object_detection([img_path])
            detected_traffic_lights = [tl[1] for tl in detected_objects[0] if tl[0] == 'traffic light']
            cols = []
            for tl in detected_traffic_lights:
                cropped_tl = get_image_crop(img, tl)
                # cv.imshow("m", cropped_tl)
                # cv.waitKey()
                # cv.destroyAllWindows()
                col = get_traffic_light_color(cropped_tl)
                if (col):
                    cols.append(col)
            if (len(set(cols))== 1): 
                print(cols[0])

            # cv.imshow(str(i), cropped_tl)
            cv.waitKey()
            cv.destroyAllWindows()

if __name__ == '__main__':
    main()