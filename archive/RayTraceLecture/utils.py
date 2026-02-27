import numpy as np
import random

def random_in_unit_sphere() -> np.ndarray:
    p = 2.0 * np.array([random.random(), random.random(), random.random()]) - np.array([1, 1, 1])
    while p.dot(p) >= 1:
        p = 2.0 * np.array([random.random(), random.random(), random.random()]) - np.array([1, 1, 1])
    return p

def reflect(v: np.ndarray, n:np.ndarray) -> np.ndarray:
    return v - 2 * np.dot(v, n) * n

def refract(v: np.ndarray, n: np.ndarray, ni_over_nt: float, refracted: np.ndarray) -> bool:
  uv: np.ndarray = v / np.linalg.norm(v)
  dt: float = np.dot(uv, n)
  discriminant: float = 1.0 - pow(ni_over_nt, 2) * (1 - pow(dt, 2))
  if discriminant > 0:
    refracted[:] = ni_over_nt * (v - n * dt) - n * np.sqrt(discriminant)
    return True
  else:
    return False

def schlick(cosine: float, ref_idx: float) -> float:
  r0: float = (1 - ref_idx) / (1 + ref_idx)
  r0 = r0 * r0
  return r0 + (1 - r0) * pow((1 - cosine), 5)

def random_in_unit_disk() -> np.ndarray:
  p: ndarray = 2 * np.array([np.random.random(), np.random.random(), 0]) - np.array([1, 1, 0])
  while np.dot(p, p) >= 1.0:
      p = 2 * np.array([np.random.random(), np.random.random(), 0]) - np.array([1, 1, 0])
  return p

def from_srgb(img_srgb):
    return np.where(img_srgb > 0.04045, ((img_srgb + 0.055) / 1.055)**2.4, img_srgb / 12.92).astype(np.float32)

def to_srgb(img):
    img_clip = np.clip(img, 0, 1)
    return np.where(img > 0.0031308, (1.055 * img_clip**(1/2.4) - 0.055), 12.92 * img_clip)

def from_srgb8(img_srgb8):
    return from_srgb(img_srgb8 / 255.0)

def to_srgb8(img):
    return np.clip(np.round(255.0 * to_srgb(img)), 0, 255).astype(np.uint8)
  
def get_sphere_uv(p: np.ndarray):
    phi: float = np.atan2(p[2], p[0])
    theta: float = np.asin(p[1])
    u = 1-(phi + np.pi) / (2*np.pi)
    v = (theta + np.pi/2) / np.pi
    return u, v
