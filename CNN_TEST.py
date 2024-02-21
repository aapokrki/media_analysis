import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from BCE_CNN import EdgeCNN

model_path = r'E:\Koulu\Vuosi4\Media Analysis\CNNtest\model\model6.pth'
model = EdgeCNN()
model.load_state_dict(torch.load(model_path))
model.eval()

# Paths to the test images
image_path = r'E:\Koulu\Vuosi4\Media Analysis\CNNtest\test\snippet_img_missing\a46492fa_snippet.tif'
mask_path = r'E:\Koulu\Vuosi4\Media Analysis\CNNtest\test\masks\a46492fa_snippet.png'
edge_map_path = r'E:\Koulu\Vuosi4\Media Analysis\CNNtest\test\snippet_edge_missing\a46492fa_snippet.tif'
gt_path = r'E:\Koulu\Vuosi4\Media Analysis\CNNtest\test\snippet_edge\a46492fa_snippet.tif'

# Load the image, mask, and edge map
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
edge_map = cv2.imread(edge_map_path, cv2.IMREAD_GRAYSCALE)
gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

# Resize the image, mask, and edge map to match the model's input size
image = cv2.resize(image, (256, 256))
mask = cv2.resize(mask, (256, 256))
edge_map = cv2.resize(edge_map, (256, 256))
gt = cv2.resize(gt, (256, 256))

# Normalize the image and mask
image = image.astype(np.float32) / 255.0
mask = mask.astype(np.float32) / 255.0
edge_map = edge_map.astype(np.float32) / 255.0
gt = gt.astype(np.float32) / 255.0

# Convert image, mask, and edge map to PyTorch tensors
image_tensor = torch.tensor(image).unsqueeze(0).unsqueeze(0)
mask_tensor = torch.tensor(mask).unsqueeze(0).unsqueeze(0)
edge_map_tensor = torch.tensor(edge_map).unsqueeze(0).unsqueeze(0)

# Pass the preprocessed image, mask, and edge map through the model
with torch.no_grad():
    output = model(image_tensor, edge_map_tensor, mask_tensor)

predicted_edge_map = output.squeeze().numpy()

threshold = 0.3

# Load the probability map (output of BCE loss)
probability_map = output  

# Binarize the probability map using the threshold
edge_map_thresholded = torch.where(probability_map > threshold, torch.tensor(1.0), torch.tensor(0.0))

# Convert thresholded edge map tensor to numpy array for visualization
edge_map_thresholded_numpy = edge_map_thresholded.squeeze().cpu().numpy()


# Visualize
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Input Image')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(mask, cmap='gray')
plt.title('Mask')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(edge_map, cmap='gray')
plt.title('Edge Map (Missing Data)')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(predicted_edge_map, cmap='gray')
plt.title('Predicted Edge Map')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(edge_map_thresholded_numpy, cmap='gray')
plt.title('Thresholded Edge Map')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(gt, cmap='gray')
plt.title('Ground Truth Edge Map')
plt.axis('off')

plt.tight_layout()
plt.show()