#!/usr/bin/env python3
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def apply_filter(image, kernel):
    """Applies a filter (kernel) to an image using convolution."""
    # Get dimensions
    image_h, image_w = image.shape
    kernel_h, kernel_w = kernel.shape

    # Determine padding size
    pad_h = kernel_h // 2
    pad_w = kernel_w // 2

    # Pad the image with zeros
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

    # Output image
    filtered_image = np.zeros_like(image , dtype=np.float32)

    # Perform convolution
    #-------Todo-----------#
    kernel_flipped =  np.flipud(np.fliplr(kernel))
    for row_in in range(pad_h, image_h + pad_h):
        for col_in in range(pad_w, image_w + pad_w):
            region = padded_image[row_in - pad_h : row_in + pad_h + 1, col_in - pad_w:col_in + pad_w + 1]
            filtered_image[row_in - pad_h, col_in - pad_w] = np.sum(region * kernel_flipped)
    # Clip values to valid range [0, 255]
    filtered_image = np.clip(filtered_image, 0, 255)

    return filtered_image
if __name__ == "__main__":
    image = cv.imread("./random_car.jpg", 0)
    if(image is None): 
        print("couldn't find the image")
        exit()
    blur_kernel = np.array([[1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1]]) / 25 
    blurred_image = apply_filter(image, blur_kernel)
    sharpen_kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1, -1]])
    sharpened_image = apply_filter(image, sharpen_kernel)
    sobel_horizontal_kernel = np.array([[-1, -2, -1],[0, 0, 0],[ 1,  2,  1]])
    edge_image = apply_filter(image, sobel_horizontal_kernel)
    blurred_image = blurred_image.astype(np.uint8)
    sharpened_image = sharpened_image.astype(np.uint8)
    edge_image = edge_image.astype(np.uint8)
    cv.imshow("original_image",image)
    cv.imshow("blurred_image",blurred_image)
    cv.imshow("sharpened_image",sharpened_image)
    cv.imshow("edge_image",edge_image)
    if cv.waitKey(0) == 27:
        cv.destroyAllWindows()
