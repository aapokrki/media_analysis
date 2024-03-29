import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import image_utils.image_processor_edge as image_processor_edge
from torch.optim.lr_scheduler import StepLR
import random
import matplotlib.pyplot as plt
import image_utils.rand_augmentation as rand_augmentation



training_image_path = r'.\data\train'
masks_path = r'.\data\test_mask\mask\testing_mask_dataset'
validation_image_path = r'.\data\validation'

# Where to save model
model_path = r'.\model\model_G1_new.pth'

# Parameters
num_epochs = 15
img_amount = 3200
val_amount = 480
batch_size = 4
learning_rate = 0.0001
betas = (0.0, 0.9)

# Training is stopped early if validation loss goes under this threshold
min_loss_threshold = 0.0001

# Parameters for adjusted BCE loss
lambda_ = 5
gamma = 2

# Parameters for learning rate scheduler
step_size_sc = 5
gamma_sc = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CUDA_LAUNCH_BLOCKING=1

# Define the Encoder
class Encoder(nn.Module):
    def __init__(self, in_channels=5):
        super(Encoder, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=0)
        self.norm1 = nn.InstanceNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2)
        self.norm2 = nn.InstanceNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2)
        self.norm3 = nn.InstanceNorm2d(256)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.norm1(self.conv1(x)))
        x = self.relu2(self.norm2(self.conv2(x)))
        x = self.relu3(self.norm3(self.conv3(x)))
        return x

# Define the Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, stride=1, dilation=2)
        self.norm1 = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out += identity

        return out
    
# Define the Decoder
class Decoder(nn.Module):
    def __init__(self, in_channels):
        super(Decoder, self).__init__()
        self.upconv1 = nn.ConvTranspose2d(in_channels, 128, kernel_size=4, stride=2, padding=1)
        self.norm1 = nn.InstanceNorm2d(128)
        self.relu1 = nn.ReLU(inplace=True)

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2,padding=0)
        self.norm2 = nn.InstanceNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        self.upconv3 = nn.ConvTranspose2d(64, 1, kernel_size=7, padding=0)
        
    def forward(self, x):
        x = self.relu1(self.norm1(self.upconv1(x)))
        x = self.relu2(self.norm2(self.upconv2(x)))
        x = self.upconv3(x)
        return x

# Define the complete CNN model
class G1(nn.Module):
    def __init__(self):
        super(G1, self).__init__()
        self.encoder = Encoder(in_channels=5)
        self.residual_blocks = nn.ModuleList([ResidualBlock(256, 256) for _ in range(8)])
        self.decoder = Decoder(in_channels=256)

    def forward(self, Im, Sm, M): 
        # Concatenate the three images along the channel dimension
        combined_input = torch.cat((Im, Sm, M), dim=1)  

        x = self.encoder(combined_input)
        for block in self.residual_blocks:
            x = block(x)
        x = self.decoder(x)
        x = torch.sigmoid(x)

        return x

class AdjustedBCELoss(nn.Module):
    def __init__(self, lambda_=5, gamma=2):
        super(AdjustedBCELoss, self).__init__()
        self.lambda_ = lambda_
        self.gamma = gamma

    def forward(self, output, target, mask):

        # Loss function as defined in the paper
        bce_loss = F.binary_cross_entropy(output, target, reduction='mean')
        adjusted_loss = ((self.lambda_*mask)*((output-target)**self.gamma))*bce_loss + \
                        (((1-mask)*(((output - target)**self.gamma)))*bce_loss)
        
        return torch.mean(adjusted_loss)
    
# Define dataset and dataloader
class EdgeDataset(Dataset):
    def __init__(self, image_path, mask_path, img_count, transform=None):
        self.image_path = image_path
        self.mask_path = mask_path
        self.transform = transform

        # Get list of file names without extensions
        self.file_names = [f.split('.')[0] for f in os.listdir(self.image_path) if f.endswith('.jpg')]
        self.file_names = self.file_names[:img_count]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]

        # Randomly choose a mask file
        mask_files = os.listdir(self.mask_path)
        mask_name = random.choice(mask_files)
        
        img_path = os.path.join(self.image_path, file_name + '.jpg')
        mask_path = os.path.join(self.mask_path, mask_name)

        Im = cv2.imread(img_path, cv2.IMREAD_COLOR)
        M = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Random crop, scaling, or flip
        rand = random.randint(1, 3)
        Im = {
            1: lambda: Im,
            2: lambda: rand_augmentation.random_crop(Im),
            #3: lambda: utils.random_scaling(Im),
            3: lambda: rand_augmentation.flip(Im)
        }[rand]()

        Im, Sm, gt_edge, M = image_processor_edge.process_image(Im, M)

        if self.transform:
            Im = self.transform(Im)
            Sm = self.transform(Sm)
            gt_edge = self.transform(gt_edge)
            M = self.transform(M)

        return Im, Sm, gt_edge, M


if __name__ == "__main__":
    model = G1()
    model.to(device)

    criterion = AdjustedBCELoss(lambda_, gamma)
    optimizer = torch.optim.Adam(model.parameters(), betas=betas, lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=step_size_sc, gamma=gamma_sc)

    transform = transforms.ToTensor()

    train_dataset = EdgeDataset(training_image_path, masks_path, img_amount, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = EdgeDataset(validation_image_path, masks_path, val_amount, transform=transform)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(num_epochs):
        
        # Training loop
        model.train()
        running_loss = 0.0
        for Im, Sm, gt_edge, M in train_dataloader:  
            optimizer.zero_grad()
            Im = Im.to(device)
            Sm = Sm.to(device)
            M = M.to(device)
            gt_edge = gt_edge.to(device)

            predicted = model(Im, Sm, M)
            loss = criterion(predicted, gt_edge, M)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * Im.size(0)

        scheduler.step()

        epoch_loss = running_loss / len(train_dataloader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}")

        # Evaluation loop
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for Im_val, Sm_val, gt_edge_val, M_val  in val_dataloader:

                Im_val = Im_val.to(device)
                Sm_val = Sm_val.to(device)
                M_val = M_val.to(device)
                gt_edge_val = gt_edge_val.to(device)

                predicted_val = model(Im_val, Sm_val, M_val)
                val_loss = criterion(predicted_val, gt_edge_val, M_val)
                val_running_loss += val_loss.item() * Im_val.size(0)

        val_epoch_loss = val_running_loss / len(val_dataloader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_epoch_loss:.6f}")

        # Break loop if the loss reaches the threshold
        if val_epoch_loss < min_loss_threshold:
            print("Validation loss is below the minimum threshold. Stopping training.")
            break

    torch.save(model.state_dict(), model_path)