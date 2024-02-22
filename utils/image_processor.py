import os
import time

import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import glob
import random

class ImageProcessor:
    def __init__(self, base_folder):
        self.base_folder = base_folder
        self.dataset_path = os.path.join(base_folder, '..','data', '2_Ortho_RGB')
        #self.dataset_files = os.listdir(self.dataset_path)
        self.dataset_files = [f for f in os.listdir(self.dataset_path) if f.endswith('.tif')]


        self.train_path = os.path.join(base_folder, '..','data', 'train')
        self.test_path = os.path.join(base_folder, '..','data', 'test')

        for folder in [self.train_path, self.test_path]:
            os.makedirs(folder, exist_ok=True)

        self.train_files = os.listdir(self.train_path)
        self.test_files = os.listdir(self.test_path)



    def random_scaling(self, image):
        scale_factor = random.uniform(0.5, 2.0)
        scaled_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)
        return scaled_image

    def horizontal_flip(self, image):
        return cv2.flip(image, 1)

    def random_crop(self, image, crop_size=(512, 512)):
        height, width, _ = image.shape
        max_y = height - crop_size[0]
        max_x = width - crop_size[1]
        start_y = random.randint(0, max_y)
        start_x = random.randint(0, max_x)
        cropped_image = image[start_y:start_y+crop_size[0], start_x:start_x+crop_size[1]]
        return cropped_image

    def display_pic(self, source_folder, image_filename):
        image_path = os.path.join(source_folder, image_filename)

        print(image_path)
        im = cv2.imread(image_path)
        print(im)
        cv2.imshow('Image', im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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
                print(f"{image_name}_{i}_{j}.jpg")
                #break  # remove this if you want to segment the entire image

            #break  # remove this if you want to segment the entire image

# Example usage
if __name__ == '__main__':
    p = ImageProcessor(os.path.dirname(__file__))

    # dataset_files = 38
    #train_files = len(p.dataset_files) / 2 #1-19
    #test_files = len(p.dataset_files) # 20-38
    train_range = range(0,18)
    test_range = range(19,37)
    i = 0

    # Generate the training images
    for i in train_range:
        p.segment_and_save(p.dataset_path, p.dataset_files[i], p.train_path, segment_size=(512, 512))
        break  # REMOVE THIS TO GENERATE THE WHOLE DATASET

    # Generate the testing images
    for i in test_range:
        p.segment_and_save(p.dataset_path, p.dataset_files[i], p.test_path, segment_size=(512, 512))
        break  # REMOVE THIS TO GENERATE THE WHOLE DATASET










