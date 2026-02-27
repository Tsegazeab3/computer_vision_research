import numpy as np

class Ray:
    def __init__(self, origin, direction, start=0., end=np.inf):
        self.origin = np.array(origin, np.float64)
        self.direction = np.array(direction, np.float64)
        self.start = start
        self.end = end