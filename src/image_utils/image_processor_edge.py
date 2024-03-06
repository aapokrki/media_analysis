import cv2 as cv
import numpy as np
import torch
import matplotlib.pyplot as plt

def create_edgemap(image):
    edgemap = cv.Canny(image, 150, 250)
    return edgemap

def create_edgemap_missing(edgemap, mask):

    edgemap_copy = np.copy(edgemap)
    edgemap_copy[mask == 255] = 0
    return edgemap_copy

def create_image_missing(image, mask):
    _, mask_img = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)
    mask_color = cv.cvtColor(mask_img, cv.COLOR_GRAY2BGR)
    mask_color = mask_color.squeeze()
    masked_image = cv.add(image, mask_color)
    return masked_image

def process_image(image, mask):

    #image = cv.resize(image, (256, 256))
    #mask = cv.resize(mask, (256, 256))

    edgemap = create_edgemap(image)
    edgemap_missing_data = create_edgemap_missing(edgemap, mask)
    masked_image = create_image_missing(image, mask)

    # Normalize
    edgemap = edgemap.astype(np.float32) / 255.0
    edgemap_missing_data = edgemap_missing_data.astype(np.float32) / 255.0
    masked_image = masked_image.astype(np.float32) / 255.0
    mask = mask.astype(np.float32) / 255.0

    return masked_image, edgemap_missing_data, edgemap, mask