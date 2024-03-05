import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PEGN_G1 import G1
from PEGN_G2 import G2
import os
import image_utils.image_processor_edge as image_processor_edge
import random

# Specify which model you want to use. G1 for BCE loss (first part of PEGN) and G2 for adversarial loss (second part of PEGN)
mode = "G1"

# Edit if you want to save the produced image
#save_image = False
#save_folder = f'.\\data\\test\\predicted_edgemaps\\{file_name}.tif'

# Edit threshold for which pixels are selected to the output
threshold = 0.4

if mode == "G1":
    model_path = r'.\model\model_G1_new.pth'
    model = G1()
elif mode == "G2":
    model_path = r'.\model\G2_test.pth'
    model = G2()

model.load_state_dict(torch.load(model_path))
model.eval()

image_folder = '.\\data\\test'
image_files = os.listdir(image_folder)
random_image_file = random.choice(image_files)
image_path = os.path.join(image_folder, random_image_file)



# Get random mask path
mask_folder = '.\\data\\test_mask\\mask\\testing_mask_dataset'
mask_files = os.listdir(mask_folder)
random_mask_file = random.choice(mask_files)
mask_path = os.path.join(mask_folder, random_mask_file)

print(image_path)
print(mask_path)

#mask_path = '.\\data\\test_mask\\mask\\testing_mask_dataset\\03988.png'
#image_path = '.\\data\\test\\top_potsdam_6_7_RGB_8_3.jpg'

# Load the image and mask
image = cv2.imread(image_path, cv2.IMREAD_COLOR)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

# Process the image and mask through the image processor
Im, Sm, gt_edge, M = image_processor_edge.process_image(image, mask)
image = cv2.resize(image, (256, 256))

# Convert processed images to PyTorch tensors
Im_tensor = torch.tensor(Im).unsqueeze(0).unsqueeze(0)
Sm_tensor = torch.tensor(Sm).unsqueeze(0).unsqueeze(0)
gt_edge_tensor = torch.tensor(gt_edge).unsqueeze(0).unsqueeze(0)
M_tensor = torch.tensor(M).unsqueeze(0).unsqueeze(0)

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
    output = model(combined_input, Sm_tensor, M_tensor)

# Calculate some visualizations
predicted_edge_map = output.squeeze().numpy()
probability_map = output  
edge_map_thresholded = torch.where(probability_map > threshold, torch.tensor(1.0), torch.tensor(0.0))
edge_map_thresholded_numpy = edge_map_thresholded.squeeze().cpu().numpy()

diff_colored = np.zeros_like(cv2.cvtColor(Sm, cv2.COLOR_GRAY2BGR))
diff = np.abs(edge_map_thresholded_numpy - Sm)
diff_colored[:, :, 1] = diff * 255

overlay = cv2.addWeighted(cv2.cvtColor(Sm, cv2.COLOR_GRAY2BGR), 0.5, diff_colored, 0.5, 0)

#if save_image:
#    thresholded_edge_map_path = os.path.join(save_folder, f'{file_name}.tif')
#    cv2.imwrite(thresholded_edge_map_path, edge_map_thresholded_numpy * 255)

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
plt.imshow(Sm, cmap='gray')
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
plt.imshow(gt_edge, cmap='gray')
plt.title('Ground Truth Edge Map')
plt.axis('off')

plt.subplot(3, 3, 7)
plt.imshow(overlay)
plt.title('Edges added by model')
plt.axis('off')

# Add more visualization plots if needed

plt.tight_layout()
plt.show()