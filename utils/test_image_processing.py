import os

import cv2
from matplotlib import pyplot as plt
from utils.image_processor import ImageProcessor

def compare_images(im1, im2):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(cv2.cvtColor(im1, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')

    axes[1].imshow(cv2.cvtColor(im2, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Edited Image')

    plt.show()

if __name__ == '__main__':
    p = ImageProcessor(os.path.dirname(__file__), '2_Ortho_RGB')

    # Read a segmented image which to test
    im1 = cv2.imread(os.path.join(p.train_path, p.train_files[4]))

    # Scale the image randomly between 0.5 and 2.0
    im2 = p.random_scaling(im1)

    # Flip the image horizontally
    im2 = p.horizontal_flip(im2)

    # Crop the image to a random size with the width and height sizes being the limits
    im2 = p.random_crop(im2, 100, 512)

    # Compare images side by side
    compare_images(im1,im2)





