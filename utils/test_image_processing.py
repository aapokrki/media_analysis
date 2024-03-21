
import cv2
from matplotlib import pyplot as plt

from utils.image_processor import ImageProcessor

'''
Demonstrates the capabilities of the ImageProcessor class by performing various image manipulations
and displaying the results using matplotlib.
'''
if __name__ == '__main__':
    p = ImageProcessor( '2_Ortho_RGB', 'test_mask')
    image_path = p.train_files[0]
    img = cv2.imread(image_path)

    # Generate masked image ("return_mask = True" also returns the used mask)
    masked_image, mask = p.apply_random_mask(img, "train", True)

    # Image manipulation
    img_scaled = p.random_scaling(img)
    img_flipped = p.flip(img)
    img_cropped = p.random_crop(img, 100, 512)

    # Show images
    fig, axes = plt.subplots(2, 3, figsize=(10, 5))

    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image')

    axes[0, 1].imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Mask')

    axes[0, 2].imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title('Masked Image')

    axes[1, 0].imshow(cv2.cvtColor(img_scaled, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('Image with random scaling')

    axes[1, 1].imshow(cv2.cvtColor(img_flipped, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('Image flipped')

    axes[1, 2].imshow(cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title('Image with random crop')

    plt.tight_layout()
    plt.show()






