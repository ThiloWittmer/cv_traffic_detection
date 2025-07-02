from map_detection import detect_turns
from pathlib import Path
from enum import Enum
import cv2 as cv
import numpy as np
from ultralytics import YOLO

#for type-hints
MatLike = np.ndarray 

yolo_model = YOLO("yolov8l.pt")

dir_imgs = Path('Project_images')
templates = [str(f) for f in Path('Project_images/templates').iterdir() if f.is_file() and f.suffix == '.png']
scenes_tmp = [[str(img) for img in scene.iterdir() if img.is_file() and img.suffix == '.png'] for scene in dir_imgs.iterdir() if scene.is_dir() and scene.name.startswith('Szene')]
stadtkarte = 'Project_images/stadtkarte/stadtplan_ohne_gebaeude.png'
scenes = {}
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
    VORF_OBEN_RECHT =   7
    VORF_UNTEN_LINKS =  8
    VORF_UNTEN_RECHTS = 9

# Schild mit parameter der Bounding box
# -> [(Sign.STOP, (50,100,20,20)), ...]
def match_templates(img: MatLike) -> list[tuple[Sign, tuple[int, int ,int , int]]]:
    print()

    #Example
    return [(Sign.STOP, (1,2,3,4))]

def object_detection(img_paths: list[str]) -> list[list[tuple[str, tuple[int, int ,int , int]]]]:
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
            x1, y1, x2, y2 = box
            x, y = x1, y1
            w, h = x2 - x1, y2 - y1
            bbox = (x, y, w, h)
            results_one_image.append((class_name, bbox))
        results_all_images.append(results_one_image)
    
    return results_all_images

def process_scene(turn:str, img_paths:list[str]) -> list[dict]:

    return []

def main():
    turns = detect_turns('Project_images/stadtkarte/stadtplan_ohne_gebaeude.png')

if __name__ == '__main__':
    main()