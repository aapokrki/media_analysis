import os
import time

import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import glob
import random

class ImageProcessor:
    def __init__(self, base_folder, dataset_folder_name, mask_folder_name):

        #Train, validation and test ratios.
        self.train = 0.6
        self.validation = 0.20
        self.test = 0.20

        #Initialize dataset filepaths
        self.base_folder = base_folder
        self.dataset_path = os.path.join(base_folder, '..','data', dataset_folder_name)
        self.dataset_files = [f for f in os.listdir(self.dataset_path) if f.endswith('.tif')]

        #Initialize train, test and validation paths
        self.train_path = os.path.join(base_folder, '..','data', 'train')
        self.test_path = os.path.join(base_folder, '..','data', 'test')
        self.validation_path = os.path.join(base_folder, '..','data', 'validation')

        #Create train, test and validation folders if needed
        for folder in [self.train_path, self.test_path, self.validation_path]:
            os.makedirs(folder, exist_ok=True)

        #Setup train, test and validation filepaths
        self.train_files = os.listdir(self.train_path)
        self.test_files = os.listdir(self.test_path)
        self.validation_files = os.listdir(self.validation_path)


        self.train_image_range = range(0, int(len(self.dataset_files) * self.train) + 1)
        self.test_image_range = range(self.train_image_range.stop + 1, self.train_image_range.stop + int(len(self.dataset_files) * self.test))
        self.validation_image_range = range(self.test_image_range.stop + 1, self.test_image_range.stop + int(len(self.dataset_files) * self.validation))



        #Initialize mask filepath list
        self.mask_path = os.path.join(base_folder, '..', 'data', mask_folder_name, 'mask', 'testing_mask_dataset')
        self.mask_files = os.listdir(self.mask_path)

        #Initialize mask ranges based on training, testing and validation ratios
        self.train_mask_length = int(len(self.mask_files) * self.train)
        self.test_mask_length = int(len(self.mask_files) * self.test)
        self.validation_mask_length = int(len(self.mask_files) * self.validation)



        # Calculate lengths of each category (2000)
        mask_category_length = len(self.mask_files) // 6

        # Calculate lengths of each segment based on ratios
        train_category_length = int(mask_category_length * self.train)
        test_category_length = int(mask_category_length * self.test)
        validation_variation_length = int(mask_category_length * self.validation)

        # Initialize lists to store segmented mask files
        train_mask_files = []
        test_mask_files = []
        validation_mask_files = []

        # Iterate over each category
        for i in range(6):
            # Calculate the starting and ending indices for the current category
            start_index = i * mask_category_length
            end_index = (i + 1) * mask_category_length

            # Slice the current category into train, test, and validation segments
            train_variation = self.mask_files[start_index:start_index + train_category_length]
            test_variation = self.mask_files[start_index + train_category_length:start_index + train_category_length + test_category_length]
            validation_variation = self.mask_files[start_index + train_category_length + test_category_length:end_index]

            # Append segmented mask files from the current category to the corresponding lists
            train_mask_files.extend(train_variation)
            test_mask_files.extend(test_variation)
            validation_mask_files.extend(validation_variation)

        # Assign segmented mask files to respective attributes
        self.train_mask_files = train_mask_files
        self.test_mask_files = test_mask_files
        self.validation_mask_files = validation_mask_files

        '''
        print("Ratios:")
        print("Train:", self.train)
        print("Test:", self.test)
        print("Validation:", self.validation)
        print()

        print("Number of dataset images:", len(self.dataset_files))
        print("Dataset image ratios:")
        print("Train:", self.train_image_range.start + 1, "-",self.train_image_range.stop +1)
        print("Test:", self.test_image_range.start + 1, "-", self.test_image_range.stop +1)
        print("Validation:", self.validation_image_range.start + 1, "-", self.validation_image_range.stop + 1)
        print()

        print("Number of mask images: ", len(self.mask_files))
        print("Mask image ratios:")
        print("Train:", self.train_mask_length)
        print("Test:", self.test_mask_length)
        print("Validation:", self.validation_mask_length)
        '''

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

    def flip(self, image, direction = 1):
        """
        Performs a flip to the given image.

        Parameters:
            image: numpy.ndarray - The input image as a NumPy array.
            direction: int - The direction of the flip (1 - horizontal, 0 - vertical, -1 - both axes)

        Returns:
            numpy.ndarray - The flipped image as a NumPy array.
        """
        return cv2.flip(image, direction)

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

    def apply_mask(self, image, mask):
        masked_image = image * mask[:, :, np.newaxis]
        return masked_image


    def apply_random_mask(self, image_path, type = "train"):

        if type == "train":
            mask_file = random.choice(self.train_mask_files)
        elif type == "test":
            mask_file = random.choice(self.test_mask_files)
        elif type == "validation":
            mask_file = random.choice(self.validation_mask_files)
        else:
            raise Exception("Invalid type")

        mask_path = os.path.join(self.mask_path, mask_file)
        mask = cv2.imread(mask_path)
        image = cv2.imread(image_path)

        masked_image = np.where(mask == 255, 255, image)

        '''
        fig, axes = plt.subplots(1, 3, figsize=(10, 5))

        axes[0].imshow(mask)
        axes[0].set_title('Mask')

        axes[1].imshow(image)
        axes[1].set_title('Original Image')

        axes[2].imshow(masked_image)
        axes[2].set_title('Masked Image')

        plt.show()
        '''


        return masked_image

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


def clear_folder(folder_path):
    # Iterate over the files in the folder
    for file_name in os.listdir(folder_path):
        # Construct the full path to the file
        file_path = os.path.join(folder_path, file_name)

        # Check if the path points to a file
        if os.path.isfile(file_path):
            # Remove the file
            os.remove(file_path)
        # If it's a directory, recursively clear its contents
        elif os.path.isdir(file_path):
            clear_folder(file_path)
            # After clearing the subdirectory, remove it
            os.rmdir(file_path)


'''
This main function clears and recreates all segmented satellite images to train, validation and test folders
'''
if __name__ == '__main__':
    dataset_folder_name = '2_Ortho_RGB'
    mask_folder_name = 'mask'

    p = ImageProcessor(os.path.dirname(__file__), dataset_folder_name, mask_folder_name)

    folders = [p.train_path,p.test_path,p.validation_path]
    for folder in folders:
        clear_folder(folder)

    # Generate the training images
    for i in p.train_image_range:
        p.segment_and_save(p.dataset_path, p.dataset_files[i], p.train_path, segment_size=(512, 512))
        break  # REMOVE THIS TO USE THE WHOLE DATASET

    # Generate the testing images
    for i in p.test_image_range:
        p.segment_and_save(p.dataset_path, p.dataset_files[i], p.test_path, segment_size=(512, 512))
        break  # REMOVE THIS TO USE THE WHOLE DATASET

    # Generate the validation images
    for i in p.validation_image_range:
        p.segment_and_save(p.dataset_path, p.dataset_files[i], p.validation_path, segment_size=(512, 512))
        break  # REMOVE THIS TO USE THE WHOLE DATASET










