from dataclasses import dataclass
from enums import Direction

@dataclass
class State:

    current_speed: int
    direction_at_next_junction: Direction
    current_direction: Direction
    zone_30: bool
    