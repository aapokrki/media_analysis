import os
import time

import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import glob
import random

class ImageProcessor:
    def __init__(self, base_folder, dataset_folder_name):
        self.base_folder = base_folder
        self.dataset_path = os.path.join(base_folder, '..','data', dataset_folder_name)
        #self.dataset_files = os.listdir(self.dataset_path)
        self.dataset_files = [f for f in os.listdir(self.dataset_path) if f.endswith('.tif')]


        self.train_path = os.path.join(base_folder, '..','data', 'train')
        self.test_path = os.path.join(base_folder, '..','data', 'test')

        for folder in [self.train_path, self.test_path]:
            os.makedirs(folder, exist_ok=True)

        self.train_files = os.listdir(self.train_path)
        self.test_files = os.listdir(self.test_path)

    def random_scaling(self, image):
        """
        Randomly scales the given image.

        Parameters:
            image: numpy.ndarray - The input image as a NumPy array.

        Returns:
            numpy.ndarray - The scaled image as a NumPy array.
        """
        scale_factor = random.uniform(0.5, 2.0)
        scaled_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)
        return scaled_image

    def horizontal_flip(self, image):
        """
        Performs horizontal flipping of the given image.

        Parameters:
            image: numpy.ndarray - The input image as a NumPy array.

        Returns:
            numpy.ndarray - The horizontally flipped image as a NumPy array.
        """
        return cv2.flip(image, 1)

    def random_crop(self, image, min_crop, max_crop):
        """
        Randomly crops the given image.

        Parameters:
            image: numpy.ndarray - The input image as a NumPy array.
            min_crop: int - The minimum size of the crop (both width and height).
            max_crop: int - The maximum size of the crop (both width and height).

        Returns:
            numpy.ndarray - The randomly cropped image as a NumPy array.
        """
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


    def segment_and_save(self, source_folder, image_filename, output_folder, segment_size=(512, 512)):
        image_path = os.path.join(source_folder, image_filename)
        image_name = image_filename.split('.')[0]

        # Read the image using OpenCV
        image = cv2.imread(image_path)

        # Get the dimensions of the original image
        height, width, _ = image.shape

        # Calculate number of segments in each dimension
        num_segments_height = height // segment_size[0]
        num_segments_width = width // segment_size[1]

        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Segment the image
        print("Generating to", output_folder)
        for i in range(num_segments_height):
            for j in range(num_segments_width):
                # Define the bounding box for the segment
                left = j * segment_size[1]
                upper = i * segment_size[0]
                right = left + segment_size[1]
                lower = upper + segment_size[0]


                # Crop the segment
                segment = image[upper:lower, left:right]

                # Save the segment
                cv2.imwrite(os.path.join(output_folder, f"{image_name}_{i}_{j}.jpg"), segment)

if __name__ == '__main__':
    p = ImageProcessor(os.path.dirname(__file__), '2_Ortho_RGB')

    train_range = range(0,18)
    test_range = range(19,37)

    # Generate the training images
    for i in train_range:
        p.segment_and_save(p.dataset_path, p.dataset_files[i], p.train_path, segment_size=(512, 512))
        break  # REMOVE THIS TO USE THE WHOLE DATASET

    # Generate the testing images
    for i in test_range:
        p.segment_and_save(p.dataset_path, p.dataset_files[i], p.test_path, segment_size=(512, 512))
        break  # REMOVE THIS TO USE THE WHOLE DATASET







