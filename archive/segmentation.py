#!/usr/bin/env python3
import cv2
import numpy as np
import matplotlib.pyplot as plt

filename = "./tennis.jpg"
img = cv2.imread(filename)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_blur = cv2.GaussianBlur(img_rgb, (11, 11), 0)
img_hsv = cv2.cvtColor(img_blur, cv2.COLOR_RGB2HSV)
edges = cv2.Canny(img_blur, 100, 200)
lower_yellow = np.array([20, 100, 100]) 
upper_yellow = np.array([35, 255, 255])
mask = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
kernel = np.ones((5,5), np.uint8)
mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
final_seg = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_clean)

plt.imshow(final_seg)
plt.title("Step 1: Raw Input (RGB Matrix)")
plt.show()

