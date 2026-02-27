import numpy as np
from Ray import Ray
from utils import random_in_unit_disk

class Camera:
    def __init__(self, eye=np.array([0,0,0]), target=np.array([0,0,-1]), up=np.array([0,1,0]), 
                 vfov=90.0, aspect=1.0, aperture: float = 10, focus_dist: float = 20):
        """
        Parameters:
          eye : (3,) -- the camera's location, aka viewpoint (a 3D point)
          target : (3,) -- where the camera is looking: a 3D point that appears centered in the view
          up : (3,) -- the camera's orientation: a 3D vector that appears straight up in the view
          vfov : float -- the full vertical field of view in degrees
          aspect : float -- the aspect ratio of the camera's view (ratio of width to height)
        """
        self.eye = eye
        self.aspect = aspect
        self.aperture = aperture
        self.focus_dist = np.linalg.norm(target - eye)
        self.local_back = (eye - target) / np.linalg.norm(eye - target)
        self.local_right = np.cross(up, self.local_back)
        self.local_right /= np.linalg.norm(self.local_right)
        self.local_up = np.cross(self.local_back, self.local_right)

        self.half_height = np.tan(np.radians(vfov) / 2)
        self.half_width = self.aspect * self.half_height

        self.lower_left_corner = (
          self.eye
          - self.focus_dist * self.local_back
          - self.half_width * self.focus_dist * self.local_right
          - self.half_height * self.focus_dist * self.local_up
        )

        self.horizontal = 2 * self.half_width * self.focus_dist * self.local_right
        self.vertical = 2 * self.half_height * self.focus_dist * self.local_up

    def generate_ray(self, img_point):
        lens_radius = self.aperture / 2
        lens_offset = lens_radius * random_in_unit_disk()
        offset = lens_offset[0] * self.local_right + lens_offset[1] * self.local_up

        target = (
          self.lower_left_corner
          + img_point[0] * self.horizontal
          + img_point[1] * self.vertical
        )

        direction = target - (self.eye + offset)

        # px = (2 * img_point[0] - 1) * self.half_width
        # py = (2 * img_point[1] - 1) * self.half_height

        # direction = -self.local_back + px * self.local_right + py * self.local_up
        direction /= np.linalg.norm(direction)

        return Ray(self.eye + offset, direction)
