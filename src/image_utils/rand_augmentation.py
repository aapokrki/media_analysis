import random
import cv2
import numpy as np

def random_scaling(image):
    scale_factor = random.uniform(0.5, 2.0)
    scaled_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)
    return scaled_image

def flip(image, direction = 1):
    return cv2.flip(image, direction)

def random_crop(image, min_crop = 100, max_crop = 512):
    height, width = image.shape[:2]

    if(max_crop > height or max_crop > width):
        max_crop = width

    # Choose a random crop size
    crop_size = np.random.randint(min_crop, min(max_crop, max_crop) + 1)

    # Randomly choose the top-left corner of the crop window
    left = np.random.randint(0, max(1, width - crop_size + 1))
    top = np.random.randint(0, max(1, height - crop_size + 1))

    # Calculate the coordinates of the bottom-right corner of the crop window
    right = min(width, left + crop_size)
    bottom = min(height, top + crop_size)

    # Crop the image
    cropped_image = image[top:bottom, left:right]

    return cropped_image