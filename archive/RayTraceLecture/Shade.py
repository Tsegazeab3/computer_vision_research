import numpy as np
from Hitable import Hit, no_hit
from Ray import Ray
import random

#DEFAULT = 4, 3
MAX_DEPTH = 4
NUMBER_OF_SAMPLES = 3

def shade(ray, hit, scene, depth = 0):
    hit = scene.intersect(ray)
    if hit.t < no_hit.t:
        scattered = Ray(np.zeros((3,)), np.zeros((3,)))
        attenuation = np.ones((3,))
        emit = hit.material.emit(0.0, 0.0, np.zeros((3,)))
        if depth < MAX_DEPTH and hit.material.scatter(ray, hit, attenuation, scattered):
            return emit + attenuation * shade(scattered, hit, scene, depth + 1)
        else:
            return emit
    return np.array([0.075, 0.3, 0.714])

def render_image(camera, scene, nx, ny):
    image = np.zeros((ny, nx, 3), np.float32)
    for j in range(ny):
        for i in range(nx):
            pixel_sample = np.zeros((3,))
            for s in range(NUMBER_OF_SAMPLES):
                #from image coordinate into texture coordinate
                img_x = (i + random.random()) / nx
                img_y = (j + random.random()) / ny
                ray = camera.generate_ray(np.array([img_x, img_y]))
                pixel_sample += shade(ray, no_hit, scene, 0)
            image[j, i] = pixel_sample / NUMBER_OF_SAMPLES
    return image