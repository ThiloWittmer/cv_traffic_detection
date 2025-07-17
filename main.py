"""
    @author:    Thilo Wittmer

    Python Anwendung, um aus Verkehrsszenen die richtigen Fahranweisungen zu berechnen

"""

from map_detection import detect_turns
from pathlib import Path
from enum import Enum
import cv2 as cv
import numpy as np
from ultralytics import YOLO
from template_matching import match_templates, sign_verification
from junction_detection_conv import junction_in_sight
from enums import Sign, Direction, TrafficColor, YoloNames
from state import State

#for type-hints
MatLike = np.ndarray 
BoundingBox = tuple[int, int, int, int]
"""[x,y,w,h]"""


yolo_model = YOLO("yolov8l.pt")

dir_imgs = Path('Project_images')
templates = [str(f) for f in Path('Project_images/templates').iterdir() if f.is_file() and f.suffix == '.png']
scenes_tmp = [[str(img) for img in scene.iterdir() if img.is_file() and img.suffix == '.png'] for scene in dir_imgs.iterdir() if scene.is_dir() and scene.name.startswith('Szene')]
stadtkarte = 'Project_images/stadtkarte/stadtplan_ohne_gebaeude.png'
scenes: dict[str,list[str]] = {}
for scene in scenes_tmp:
    k = scene[0].split('/')[1]
    scene = sorted(scene, key=lambda x: int(x.split('/')[-1].split('.')[0]))
    scenes[k] = scene
current_speed_limit = 50
result = {}

state = State(
    current_speed=50,
    direction_at_next_junction=Direction.GERADEAUS,
    current_direction=Direction.GERADEAUS,
    zone_30=False,
)

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

def car_is_infront() -> bool:
    #TODO implement
    return False

def car_is_left() -> bool:
    #TODO implement

    return False

def process_scene(turn:Direction, img_paths:list[str]) -> list[dict] | None:
    """returns list of dict where each dict represents one image in the scene\n
        each dict has the following format:\n
        {
            \"bildname\":           \"0.png\",
            \"geschwindigkeit\":    50,
            \"richtung\":           \"links\"
        }
    """

    results_scene = []

    state.direction_at_next_junction = turn

    state.current_direction = Direction.GERADEAUS

    if state.zone_30:
        state.current_speed = 30
    else:
        state.current_speed = 50
    
    #liste, jedes element enthält ergebnisse für ein bild
    detected_objects = object_detection(img_paths)

    #hier jedes Bild durchgehen
    for i, det_obj in enumerate(detected_objects):
        print(f"### {i} ###")
        img: MatLike | None = cv.imread(img_paths[i])

        if img is None:
            raise IOError("Bild konnte nicht gelesen werden")
        
        ampel_det: list[tuple[str, BoundingBox]] = []
        fahrzeug_det: list[tuple[str, BoundingBox]] = []
        stop_det: list[tuple[str, BoundingBox]] = []

        for det in det_obj:
            match(det[0]):
                case YoloNames.AMPEL.value:
                    ampel_det.append(det)
                case YoloNames.STOP.value:
                    stop_det.append(det)
                case YoloNames.AUTO.value | YoloNames.LKW.value | YoloNames.MOTORRAD.value | YoloNames.BUS.value:
                    fahrzeug_det.append(det)
        

        print(f"Erkannte fahrzeuge: {[f[0] for f in fahrzeug_det]}")
        print(f"Erkannte ampeln: {len(ampel_det)}")
        print(f"Erkannte stopschilder: {len(stop_det)}")

        #ampeln erkannt
        if ampel_det:
            ampel_gruen = handle_traffic_light(img, ampel_det)
            
            if ampel_gruen:

                match(turn):
                    case Direction.LINKS:
                        if car_is_left():
                            state.current_speed = 0
                    case Direction.RECHTS:
                        state.current_speed = 10
                    case Direction.GERADEAUS:
                        
                        print()

    return None

def handle_traffic_light(img, ampel_det) -> bool:
    #threshhold for slowing down    
    AMPEL_AREA_THRESHHOLD_S = 2000
    #threshhold for stopping
    AMPEL_AREA_THRESHHOLD_L = 15000

    ampel_gruen = False
    state.current_direction = state.direction_at_next_junction
    farben: list[TrafficColor] = []
    ampel_groesse: list[int] = []
    for ampel in ampel_det:
        farbe: TrafficColor | None = get_traffic_light_color(get_image_crop(img, ampel[1]))
                #bb w*h
        flaeche = int(ampel[1][2]*ampel[1][3])
        ampel_groesse.append(flaeche)
        if farbe:
            farben.append(farbe)
            
    aktuelle_ampelfarbe: TrafficColor
    groesste_ampel: int = max(ampel_groesse)


    if not len(set(farben)) == 1:
        print("Ampelfarben mismatch")
        farben_count: dict[TrafficColor, int] = {}
        for farbe in farben:
            if not farbe in farben_count.keys():
                farben_count[farbe] = 1
            else:
                farben_count[farbe] += 1
        aktuelle_ampelfarbe = max(farben_count, key=lambda k: farben_count[k])
    else:
        aktuelle_ampelfarbe = set(farben).pop()
            
    match(aktuelle_ampelfarbe):
        case TrafficColor.RED | TrafficColor.RED_YELLOW | TrafficColor.YELLOW:
            if groesste_ampel > AMPEL_AREA_THRESHHOLD_S and groesste_ampel < AMPEL_AREA_THRESHHOLD_L:
                state.current_speed = 10
            if groesste_ampel > AMPEL_AREA_THRESHHOLD_L:
                state.current_speed = 0
        case TrafficColor.GREEN:
            ampel_gruen = True
    return ampel_gruen


def main():
    process_scene(turn= Direction.LINKS, img_paths=scenes['Szene_3'])

    # TEST template matching
    # imgs = scenes.get("Szene_5")
    # if imgs:
    #     img = imgs[4]
    #     test = "Project_images/templates/stop.png"
    #     img = cv.imread(img)
    #     if img is not None:
    #         cv.imshow("test", img)
    #         cv.waitKey()
    #         cv.destroyAllWindows()
    #         w, h = img.shape[:2] 
    #         for match in match_templates(get_image_crop(img, (int(w/2), 0, h,w))):
    #             print(f"{match[0]}: {match[1]}")
    #     else:
    #         print(f"Failed to load image: {imgs[4]}")
    
    # TEST Kreuzung erkennen
    # for k, imgs in scenes.items():
    #     if imgs: 
    #         for i, img in enumerate(imgs):
    #             img = cv.imread(img)    
    #             print(f"{i}: {junction_in_sight(img, visualize=True)}")

    # img = cv.imread("Project_images/templates/vorfahrt.png")
    # cv.imshow('org', img)
    # cv.waitKey()
    # cv.destroyAllWindows()
    
    # print(sign_verification(img, sign = Sign.VORF))

    # TEST ampel farbe erkennen
    # scene = scenes.get("Szene_3")
    # if scene:
    #     for img_path in scene:
    #         img = cv.imread(img_path)
    #         cv.imshow("test", img)
    #         detected_objects = object_detection([img_path])
    #         detected_traffic_lights = [tl[1] for tl in detected_objects[0] if tl[0] == 'traffic light']
    #         cols = []
    #         for tl in detected_traffic_lights:
    #             cropped_tl = get_image_crop(img, tl)
    #             # cv.imshow("m", cropped_tl)
    #             # cv.waitKey()
    #             # cv.destroyAllWindows()
    #             col = get_traffic_light_color(cropped_tl)
    #             if (col):
    #                 cols.append(col)
    #         if (len(set(cols))== 1): 
    #             print(cols[0])

    #         # cv.imshow(str(i), cropped_tl)
    #         cv.waitKey()
    #         cv.destroyAllWindows()

if __name__ == '__main__':
    main()