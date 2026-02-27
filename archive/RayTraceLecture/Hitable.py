import numpy as np
from Ray import *

class Hit:
    def __init__(self, t, point=None, normal=None, material=None):
        self.t = t
        self.point = point
        self.normal = normal
        self.material = material

no_hit = Hit(np.inf)

class Hitable:
    def __init__(self, material = None):
        self.material = material
    def intersect(self, ray):
        pass

class Scene(Hitable):
    def __init__(self, surfs, bg_color=np.array([0.2, 0.3, 0.5])):
        self.surfs = surfs
        self.bg_color = bg_color
    def intersect(self, ray):
        closest_hit = no_hit
        for surf in self.surfs:
            hit = surf.intersect(ray)
            if hit.t < closest_hit.t:
                closest_hit = hit
        
        return closest_hit

class Sphere(Hitable):
    def __init__(self, center, radius, material):
        super().__init__(material = material)
        self.center = center
        self.radius = radius
    def intersect(self, ray):
        # pre-compute
        r_sq = pow(self.radius, 2)

        l = self.center - ray.origin
        s = l.dot(ray.direction)
        l_sq = l.dot(l)
        if s < 0 and l_sq > r_sq:
          return no_hit
        m_sq = l_sq - pow(s, 2)
        if m_sq > r_sq:
          return no_hit
        q = np.sqrt(r_sq - m_sq)
        if l_sq > r_sq:
          t = s - q
        else:
          t = s + q
        ray_direction_normalized = np.divide(ray.direction, np.linalg.norm(ray.direction))
        point = ray.origin + t * ray_direction_normalized
        surface_normal = point - self.center
        return Hit(t, point, np.divide(surface_normal,np.linalg.norm(surface_normal)), self.material)

class XYPlane(Hitable):
  def __init__(self, x0: float, x1: float, y0: float, y1: float, k: float, material):
    super().__init__(material=material)
    self.x0 = x0
    self.x1 = x1
    self.y0 = y0
    self.y1 = y1
    self.k = k
  def intersect(self, ray, t0: float = 0, t1: float = np.inf):
    """
    z(t) = k
    k = oz + tdz
    t = (k - oz) / dz
    x = ox + tdx
    y = oy + tdy
    t <= t0 && t >= t1
    x <= x0 && x >= x1
    y <= y0 && y >= y1
    """
    t: float = (self.k - ray.origin[2]) / ray.direction[2]
    if t <= t0 or t >= t1:
      return no_hit
    x: float = ray.origin[0] + t * ray.direction[0]
    y: float = ray.origin[1] + t * ray.direction[1]
    if x <= self.x0 or x >= self.x1 or y <= self.y0 or y >= self.y1:
      return no_hit
    
    normal: np.ndarray = np.array([0, 0, 1])
    point: np.ndarray = ray.origin + t * ray.direction
    return Hit(t, point, normal, self.material)

class XZPlane(Hitable):
  def __init__(self, x0: float, x1: float, z0: float, z1: float, k: float, material):
    super().__init__(material=material)
    self.x0 = x0
    self.x1 = x1
    self.z0 = z0
    self.z1 = z1
    self.k = k
  def intersect(self, ray, t0: float = 0, t1: float = np.inf):
    """
    y(t) = k
    k = oy + tdy
    t = (k - oy) / dy
    x = ox + tdx
    z = oz + tdz
    t <= t0 && t >= t1
    x <= x0 && x >= x1
    z <= z0 && z >= z1
    """
    t: float = (self.k - ray.origin[1]) / ray.direction[1]
    if t <= t0 or t >= t1:
      return no_hit
    x: float = ray.origin[0] + t * ray.direction[0]
    z: float = ray.origin[2] + t * ray.direction[2]
    if x <= self.x0 or x >= self.x1 or z <= self.z0 or z >= self.z1:
      return no_hit
    
    normal: np.ndarray = np.array([0, 1, 0])
    point: np.ndarray = ray.origin + t * ray.direction
    return Hit(t, point, normal, self.material)

class YZPlane(Hitable):
  def __init__(self, y0: float, y1: float, z0: float, z1: float, k: float, material):
    super().__init__(material=material)
    self.y0 = y0
    self.y1 = y1
    self.z0 = z0
    self.z1 = z1
    self.k = k
  def intersect(self, ray, t0: float = 0, t1: float = np.inf):
    """
    x(t) = k
    k = ox + tdx
    t = (k - ox) / dx
    y = oy + tdy
    z = oz + tdz
    t <= t0 && t >= t1
    y <= y0 && y >= y1
    z <= z0 && z >= z1
    """
    t: float = (self.k - ray.origin[0]) / ray.direction[0]
    if t <= t0 or t >= t1:
      return no_hit
    y: float = ray.origin[1] + t * ray.direction[1]
    z: float = ray.origin[2] + t * ray.direction[2]
    if y <= self.y0 or y >= self.y1 or z <= self.z0 or z >= self.z1:
      return no_hit
    
    normal: np.ndarray = np.array([1, 0, 0])
    point: np.ndarray = ray.origin + t * ray.direction
    return Hit(t, point, normal, self.material)

class FlipPlane(Hitable):
  def __init__(self, plane):
    super().__init__(material = plane.material)
    self.plane = plane
  def intersect(self, ray, t0: float = 0, t1: float = np.inf):
    hit = self.plane.intersect(ray)
    if hit.t < no_hit.t:
      hit.normal = -hit.normal
    return hit

class Box(Hitable):
  def __init__(self, p0: np.ndarray, p1: np.ndarray, material):
    super().__init__(material=material)
    self.pmin = p0
    self.pmax = p1
    self.plane_list = [
      XYPlane(p0[0], p1[0], p0[1], p1[1], p1[2], self.material),
      FlipPlane(XYPlane(p0[0], p1[0], p0[1], p1[1], p0[2], self.material)),
      XZPlane(p0[0], p1[0], p0[2], p1[2], p1[1], self.material),
      FlipPlane(XZPlane(p0[0], p1[0], p0[2], p1[2], p0[1], self.material)),
      YZPlane(p0[1], p1[1], p0[2], p1[2], p1[0], self.material),
      FlipPlane(YZPlane(p0[1], p1[1], p0[2], p1[2], p0[0], self.material)),
    ]
  def intersect(self, ray, t0: float = 0, t1: float = np.inf):
    closest_hit = no_hit
    for plane in self.plane_list:
      hit = plane.intersect(ray)
      if hit.t < closest_hit.t:
        closest_hit = hit
    return closest_hit

class Translate(Hitable):
  def __init__(self, p, displacement: np.ndarray):
    self.ptr = p
    self.offset = displacement
  def intersect(self, ray, t0: float = 0, t1: float = np.inf):
    moved_ray = Ray(ray.origin - self.offset, ray.direction)
    hit = self.ptr.intersect(moved_ray, t0, t1)
    if hit.t < no_hit.t:
      hit.point += self.offset
    return hit

class RotateY(Hitable):
  def __init__(self, p, angle):
    self.ptr = p
    self.radian = (np.pi / 180) * angle
    self.sin_theta = np.sin(self.radian)
    self.cos_theta = np.cos(self.radian)
  def intersect(self, ray, t0: float = 0, t1: float = np.inf):
    origin: np.ndarray = ray.origin
    direction: np.ndarray = ray.direction
    origin[0] = self.cos_theta * ray.origin[0] - self.sin_theta * ray.origin[2]
    origin[2] = self.sin_theta * ray.origin[0] + self.cos_theta * ray.origin[2]
    direction[0] = self.cos_theta * ray.direction[0] - self.sin_theta * ray.direction[2]
    direction[2] = self.sin_theta * ray.direction[0] + self.cos_theta * ray.direction[2]
    rotated_r = Ray(origin, direction)
    hit = self.ptr.intersect(rotated_r, t0, t1)
    if hit.t < no_hit.t:
      p: np.ndarray = hit.point
      normal: np.ndarray = hit.normal
      p[0] = self.cos_theta * hit.point[0] + self.sin_theta * hit.point[2]
      p[2] = -self.sin_theta * hit.point[0] + self.cos_theta * hit.point[2]
      normal[0] = self.cos_theta * hit.normal[0] + self.sin_theta * hit.normal[2]
      normal[2] = -self.sin_theta * hit.normal[0] + self.cos_theta * hit.normal[2]
      hit.point = p
      hit.normal = normal
    
    return hit