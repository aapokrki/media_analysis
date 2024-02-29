import os
import numpy as np
import cv2
import random

'''
This class ImageProcessor initializes the necessary filepaths for images and masks and allows the user to manipulate
and use masks on any image. It also specifies the train, validation and test ratios. The ratios will be used in order.
e.g. From the dataset, the first 60% will be training data images and the next 20% will be the validation images and so forth.  

To regenerate the generated segmented images with new train,validate and test ratios, run the main-function of this file.

Initialize the ImageProcessor class with the correct filenames of the dataset_folder and mask_folder respectively.

'''


class ImageProcessor:
    def __init__(self, dataset_folder_name, mask_folder_name):
        """
           Initialize an ImageProcessor object with dataset and mask folder names.

           Args:
               dataset_folder_name (str): The name of the folder containing the dataset (in ./data/).
               mask_folder_name (str): The name of the folder containing the masks (in ./data/).

           Attributes:
               train (float): The ratio of images used for training.
               validation (float): The ratio of images used for validation.
               test (float): The ratio of images used for testing.
               base_folder (str): The base folder path.
               dataset_path (str): The path to the dataset folder.
               dataset_files (list): List of filenames in the dataset folder.
               train_path (str): The path to the training dataset folder.
               test_path (str): The path to the testing dataset folder.
               validation_path (str): The path to the validation dataset folder.
               train_files (list): List of filepaths in the training dataset folder.
               test_files (list): List of filepaths in the testing dataset folder.
               validation_files (list): List of filepaths in the validation dataset folder.
               mask_path (str): The path to the mask folder.
               mask_files (list): List of filenames in the mask folder.
               train_mask_files (list): List of filepaths in the training mask folder.
               test_mask_files (list): List of filepaths in the testing mask folder.
               validation_mask_files (list): List of filepaths in the validation mask folder.
           """


        #Train, validation and test ratios.
        self.train = 0.7
        self.validation = 0.15
        self.test = 0.15

        #Initialize dataset filepaths
        self.base_folder = os.path.dirname(__file__)
        self.dataset_path = os.path.join(self.base_folder, '..','data', dataset_folder_name)
        self.dataset_files = [f for f in os.listdir(self.dataset_path) if f.endswith('.tif')]

        #Initialize train, test and validation paths
        self.train_path = os.path.join(self.base_folder, '..','data', 'train')
        self.test_path = os.path.join(self.base_folder, '..','data', 'test')
        self.validation_path = os.path.join(self.base_folder, '..','data', 'validation')

        #Create train, test and validation folders if needed
        for folder in [self.train_path, self.test_path, self.validation_path]:
            os.makedirs(folder, exist_ok=True)

        #Setup train, test and validation filepaths
        self.train_files = [os.path.join(self.train_path, train_file) for train_file in os.listdir(self.train_path)]
        self.test_files = [os.path.join(self.test_path, test_file) for test_file in os.listdir(self.test_path)]
        self.validation_files = [os.path.join(self.validation_path, validation_file) for validation_file in os.listdir(self.validation_path)]

        #Dataset ranges based on train, validation and test ratios, used in generating segmented images
        self.train_image_range = range(0, int(len(self.dataset_files) * self.train) + 1)
        self.test_image_range = range(self.train_image_range.stop + 1, self.train_image_range.stop + int(len(self.dataset_files) * self.test))
        self.validation_image_range = range(self.test_image_range.stop + 1, self.test_image_range.stop + int(len(self.dataset_files) * self.validation))

        #Initialize mask filepath list
        self.mask_path = os.path.join(self.base_folder, '..', 'data', mask_folder_name, 'mask', 'testing_mask_dataset')
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

        # Assign segmented mask filepaths to respective attributes
        self.train_mask_files = [os.path.join(self.mask_path, train_mask_file) for train_mask_file in train_mask_files]
        self.test_mask_files = [os.path.join(self.mask_path, test_mask_file) for test_mask_file in test_mask_files]
        self.validation_mask_files = [os.path.join(self.mask_path, validation_mask_file) for validation_mask_file in validation_mask_files]

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

    def random_crop(self, image, min_crop = 100, max_crop = 512):
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

    def apply_mask(self, image, mask, return_mask=False):
        """
        Apply a mask to the given image.

        Args:
            image (numpy.ndarray): The input image to be masked.
            mask (numpy.ndarray): The mask to be applied to the image.
            return_mask (bool, optional): Whether to return the applied mask along with the masked image. Defaults to False.

        Returns:
            numpy.ndarray or tuple: The masked image if return_mask is False, otherwise a tuple containing the masked image and the applied mask.
        """
        # Resize the mask only if it's a different size from the image
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

        # Apply the mask on the image
        masked_image = np.where(mask == 255, 255, image)

        if return_mask:
            return masked_image, mask

        return masked_image

    def apply_random_mask(self, image, mask_type="train", return_mask=False):
        """
        Apply a randomly selected mask from the specified type to the given image.

        Args:
            image (numpy.ndarray): The input image to be masked.
            mask_type (str, optional): The type of mask to be applied. Defaults to "train".
            return_mask (bool, optional): Whether to return the applied mask along with the masked image. Defaults to False.

        Raises:
            Exception: Raised if an invalid mask type is provided.

        Returns:
            numpy.ndarray or tuple: The masked image if return_mask is False, otherwise a tuple containing the masked image and the applied mask.
        """
        if mask_type == "train":
            mask_file = random.choice(self.train_mask_files)
        elif mask_type == "test":
            mask_file = random.choice(self.test_mask_files)
        elif mask_type == "validation":
            mask_file = random.choice(self.validation_mask_files)
        else:
            raise Exception("Invalid mask type")

        mask_path = os.path.join(self.mask_path, mask_file)
        mask = cv2.imread(mask_path)

        # Resize the mask only if it's a different size from the image
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

        # Apply the mask on the image
        masked_image = np.where(mask == 255, 255, image)

        if return_mask:
            return masked_image, mask

        return masked_image

    def segment_and_save(self, source_folder, image_filename, output_folder, segment_size=(512, 512)):
        """
        Segment an image into smaller segments and save them to the specified output folder.

        Args:
            source_folder (str): The folder containing the source image.
            image_filename (str): The filename of the source image.
            output_folder (str): The folder where segmented images will be saved.
            segment_size (tuple, optional): The size of each segment. Defaults to (512, 512).
        """
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
        print("Segmenting", image_name ,"to", output_folder)
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
    """
    Clear the contents of the specified folder.

    Args:
        folder_path (str): The path to the folder to be cleared.
    """
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
    mask_folder_name = 'test_mask'

    p = ImageProcessor( dataset_folder_name, mask_folder_name)

    #Clear previous segmented images to make room for new ones
    folders = [p.train_path,p.test_path,p.validation_path]
    for folder in folders:
        clear_folder(folder)

    # Generate the training images
    for i in p.train_image_range:
        p.segment_and_save(p.dataset_path, p.dataset_files[i], p.train_path, segment_size=(512, 512))
        #break  # REMOVE THIS TO USE THE WHOLE DATASET

    # Generate the testing images
    for i in p.test_image_range:
        p.segment_and_save(p.dataset_path, p.dataset_files[i], p.test_path, segment_size=(512, 512))
        #break  # REMOVE THIS TO USE THE WHOLE DATASET

    # Generate the validation images
    for i in p.validation_image_range:
        p.segment_and_save(p.dataset_path, p.dataset_files[i], p.validation_path, segment_size=(512, 512))
        #break  # REMOVE THIS TO USE THE WHOLE DATASET


    print("Success!")







