import numpy as np
from Ray import Ray
from Hitable import Hit
from utils import *
from Texture import *

class Material:
    def __init__(self, k_d):
        """
            k_d : diffuse coefficient
        """
        self.k_d = k_d
    def scatter(self, ray_in: Ray, hit: Hit, attenuation: np.ndarray, scattered: Ray):
        pass
    def emit(self, u: float, v: float, p: np.ndarray) -> np.ndarray:
      return np.zeros((3,))

class Lambertian(Material):
    def __init__(self, k_d):
        super().__init__(k_d = k_d)
    def scatter(self, ray_in: Ray, hit: Hit, attenuation: np.ndarray, scattered: Ray):
        target = hit.point + hit.normal + random_in_unit_sphere()
        scattered.origin = hit.point
        scattered.direction = target - hit.point
        u, v = get_sphere_uv((hit.point - np.array([0.7,0,0])) / np.linalg.norm((hit.point - np.array([0.7,0,0]))))
        attenuation[:] = self.k_d.value(u, v, hit.point)
        return True

class Metal(Material):
  def __init__(self, k_d: np.ndarray):
      super().__init__(k_d = k_d)
  def scatter(self, ray_in: 'Ray', hit: 'Hit', attenuation: np.ndarray, scattered: 'Ray') -> bool:
      reflected: np.ndarray = reflect(ray_in.direction / np.linalg.norm(ray_in.direction), hit.normal)
      scattered.origin = hit.point
      scattered.direction = reflected
      attenuation[:] = self.k_d.value(0, 0, hit.point)
      return np.dot(scattered.direction, hit.normal) > 0

class Dielectric(Material):
  def __init__(self, ri: float):
    self.ref_idx = ri
  def scatter(self, ray_in: 'Ray', hit: 'Hit', attenuation: np.ndarray, scattered: 'Ray') -> bool:
    outward_normal = np.zeros((3,))
    reflected: np.ndarray = reflect(ray_in.direction, hit.normal)
    ni_over_nt: float = 0.0
    attenuation[:] = np.array([1.0, 1.0, 1.0])
    refracted: np.ndarray = np.zeros((3,))
    reflect_prob: float = 0.0
    cosine: float = 0.0

    if(np.dot(ray_in.direction, hit.normal) > 0):
      outward_normal = -hit.normal
      ni_over_nt = self.ref_idx

    else:
      outward_normal = hit.normal
      ni_over_nt = 1.0 / self.ref_idx
    
    if refract(ray_in.direction, outward_normal, ni_over_nt, refracted):
      scattered.origin = hit.point
      scattered.direction = refracted
    else:
      scattered.origin = hit.point
      scattered.direction = reflected
      # return False
    
    return True