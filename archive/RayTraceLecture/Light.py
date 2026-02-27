import numpy as np
from Material import Material
from Ray import Ray
from Hitable import Hit

class DiffuseLight(Material):
    def __init__(self, k_d):
        super().__init__(k_d = k_d)
    def scatter(self, ray_in: Ray, hit: Hit, attenuation: np.ndarray, scattered: Ray):
        return False
    def emit(self, u: float, v: float, p: np.ndarray):
        return self.k_d.value(u, v, p)