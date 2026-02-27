#!/usr/bin/env python3
import numpy as np
from PIL import Image
from Hitable import Scene, Sphere
from Material import Lambertian, Metal
from cli import render
from Camera import Camera
from Texture import *
from Light import *

metallic_texture = ConstantTexture(np.array([0.9, 0.9, 0.9]))
blue_texture = ConstantTexture(np.array([0.2, 0.2, 0.5]))
gray_texture = ConstantTexture(np.array([0.5, 0.5, 0.5]))
diffuse_light_texture = ConstantTexture(np.array([4.0, 4.0, 4.0]))

metal = Metal(metallic_texture)
blue = Lambertian(blue_texture)
gray = Lambertian(gray_texture)
light = DiffuseLight(diffuse_light_texture)

scene = Scene([
    Sphere(np.array([-0.7,0,0]), 1, metal),
    Sphere(np.array([0.7,0,0]), 0.5, blue),
    Sphere(np.array([0, -40, 0]), 39.5, gray),
    Sphere(np.array([0.2, 0.7, 1]), 0.1, light),
])

camera = Camera(np.array([3,1.2,5]), target=np.array([-0.5,0,0]), vfov=24, aspect=16/9, aperture=0.0)

render(camera, scene)
