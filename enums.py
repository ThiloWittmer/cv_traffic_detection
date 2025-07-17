from enum import Enum

class Direction(Enum):
    GERADEAUS = "geradeaus"
    RECHTS = "rechts"
    LINKS = "links"

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


class YoloNames(Enum):
    AMPEL       = "traffic light"
    AUTO        = "car"
    LKW         = "truck"
    MOTORRAD    = "motorcycle"
    BUS         = "bus"
    STOP        = "stop sign"