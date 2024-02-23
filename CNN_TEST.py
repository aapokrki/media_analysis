import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from BCE_CNN import EdgeCNN

model_path = r'.\model\model8.pth'
model = EdgeCNN()
model.load_state_dict(torch.load(model_path))
model.eval()


# Path to the test image (assuming the image, mask, edge map and ground truth are named the same)
file_name = "a46492fa_snippet"

image_path = f'.\\data\\test\\snippet_img_missing\\{file_name}.tif'
mask_path = f'.\\data\\test\masks\\{file_name}.png'
edge_map_path = f'.\\data\\test\\snippet_edge_missing\\{file_name}.tif'
gt_path = f'.\\data\\test\\snippet_edge\\{file_name}.tif'

threshold = 0.5

# Load the image, mask, and edge map
image = cv2.imread(image_path, cv2.IMREAD_COLOR)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
edge_map = cv2.imread(edge_map_path, cv2.IMREAD_GRAYSCALE)
gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

# Normalize the image and mask
image = image.astype(np.float32) / 255.0
mask = mask.astype(np.float32) / 255.0
edge_map = edge_map.astype(np.float32) / 255.0
gt = gt.astype(np.float32) / 255.0

# Convert image, mask, and edge map to PyTorch tensors
image_tensor = torch.tensor(image).unsqueeze(0).unsqueeze(0)
mask_tensor = torch.tensor(mask).unsqueeze(0).unsqueeze(0)
edge_map_tensor = torch.tensor(edge_map).unsqueeze(0).unsqueeze(0)

R_channel, G_channel, B_channel = cv2.split(image)
R_channel = R_channel.astype(np.float32) / 255.0
G_channel = G_channel.astype(np.float32) / 255.0
B_channel = B_channel.astype(np.float32) / 255.0
R_tensor = torch.tensor(R_channel).unsqueeze(0).unsqueeze(0)
G_tensor = torch.tensor(G_channel).unsqueeze(0).unsqueeze(0)
B_tensor = torch.tensor(B_channel).unsqueeze(0).unsqueeze(0)
R_tensor = torch.nn.functional.interpolate(R_tensor, size=(256, 256), mode='nearest')
G_tensor = torch.nn.functional.interpolate(G_tensor, size=(256, 256), mode='nearest')
B_tensor = torch.nn.functional.interpolate(B_tensor, size=(256, 256), mode='nearest')

# Send inputs to the model
combined_input = torch.cat((R_tensor, G_tensor, B_tensor), dim=1)
with torch.no_grad():
    output = model(combined_input, edge_map_tensor, mask_tensor)


# Calculate some visualizations
predicted_edge_map = output.squeeze().numpy()

probability_map = output  
edge_map_thresholded = torch.where(probability_map > threshold, torch.tensor(1.0), torch.tensor(0.0))
edge_map_thresholded_numpy = edge_map_thresholded.squeeze().cpu().numpy()
diff = np.abs(edge_map_thresholded_numpy - edge_map)
diff_colored = np.zeros_like(image)
diff_colored[:,:,1] = diff * 255
overlay = cv2.addWeighted(cv2.cvtColor(edge_map, cv2.COLOR_GRAY2BGR), 0.5, diff_colored, 0.5, 0)

# Visualize
plt.figure(figsize=(15, 15))

plt.subplot(3, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Input Image')
plt.axis('off')

plt.subplot(3, 3, 2)
plt.imshow(mask, cmap='gray')
plt.title('Mask')
plt.axis('off')

plt.subplot(3, 3, 3)
plt.imshow(edge_map, cmap='gray')
plt.title('Edge Map (Missing Data)')
plt.axis('off')

plt.subplot(3, 3, 4)
plt.imshow(predicted_edge_map, cmap='gray')
plt.title('Predicted Edge Map')
plt.axis('off')

plt.subplot(3, 3, 5)
plt.imshow(edge_map_thresholded_numpy, cmap='gray')
plt.title('Thresholded Edge Map')
plt.axis('off')

plt.subplot(3, 3, 6)
plt.imshow(gt, cmap='gray')
plt.title('Ground Truth Edge Map')
plt.axis('off')

plt.subplot(3, 3, 7)
plt.imshow(overlay)
plt.title('Edges added by model')
plt.axis('off')

plt.tight_layout()
plt.show()
