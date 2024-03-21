import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.utils.spectral_norm as spectral_norm
import random
import image_utils.image_processor_edge as image_processor_edge
import image_utils.rand_augmentation as rand_augmentation
import matplotlib.pyplot as plt



training_image_path = r'.\data\train'
masks_path = r'.\data\test_mask\mask\testing_mask_dataset'
validation_image_path = r'.\data\validation'

# Where to save model
model_path = r'.\model\G2_test_bce.pth'

# Parameters
num_epochs = 30
img_amount = 3200
val_amount = 320
batch_size = 4
learning_rate = 0.0001
d_learning_rate = 0.00001
betas = (0.0, 0.9)

# Training is stopped early if validation loss goes under this threshold
min_loss_threshold = 0.0001

# Parameters for learning rate scheduler
step_size_sc = 20
gamma_sc = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CUDA_LAUNCH_BLOCKING=1

# Define the Encoder
class Encoder(nn.Module):
    def __init__(self, in_channels=5):
        super(Encoder, self).__init__()
        self.conv1 = spectral_norm(nn.Conv2d(in_channels, 64, kernel_size=3, padding=0, stride=2))
        self.in1 = nn.InstanceNorm2d(num_features=64)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2))
        self.in2 = nn.InstanceNorm2d(num_features=128)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2))
        self.in3 = nn.InstanceNorm2d(num_features=256)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.in1((self.conv1(x))))
        x = self.relu2(self.in2((self.conv2(x))))
        x = self.relu3(self.in3((self.conv3(x))))
        return x

# Define the Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = spectral_norm(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=2, stride=1, dilation=2))
        self.in1 = nn.InstanceNorm2d(num_features=256)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = spectral_norm(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))
        self.in2 = nn.InstanceNorm2d(num_features=256)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.in1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.in2(out)
        out += identity
        return out
    
# Define the Decoder
class Decoder(nn.Module):
    def __init__(self, in_channels):
        super(Decoder, self).__init__()
        self.upconv1 = spectral_norm(nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=0))
        self.in1 = nn.InstanceNorm2d(num_features=128)
        self.relu1 = nn.ReLU(inplace=True)

        self.upconv2 = spectral_norm(nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=0))
        self.in2 = nn.InstanceNorm2d(num_features=64)
        self.relu2 = nn.ReLU(inplace=True)

        self.upconv3 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)
        self.in3 = nn.InstanceNorm2d(num_features=1)
        self.relu3 = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.relu1(self.in1((self.upconv1(x))))
        x = self.relu2(self.in2((self.upconv2(x))))
        x = self.upconv3(x)
        return x

# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self, in_channels=4):
        super(Discriminator, self).__init__()
        self.conv1 = spectral_norm(nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1))
        self.conv2 = spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1))
        self.conv3 = spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1))
        self.conv4 = spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=0))
        self.conv5 = spectral_norm(nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0))

    def forward(self, x):
        conv1 = F.leaky_relu(self.conv1(x), negative_slope=0.2, inplace=True)
        conv2 = F.leaky_relu(self.conv2(conv1), negative_slope=0.2, inplace=True)
        conv3 = F.leaky_relu(self.conv3(conv2), negative_slope=0.2, inplace=True)
        conv4 = F.leaky_relu(self.conv4(conv3), negative_slope=0.2, inplace=True)
        conv5 = self.conv5(conv4)
        #outputs = conv5
        outputs = torch.sigmoid(conv5)
        return outputs, [conv1, conv2, conv3, conv4, conv5]

# Define the complete CNN model
class G2(nn.Module):
    def __init__(self):
        super(G2, self).__init__()
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

class AdversarialLoss(nn.Module):
    def __init__(self):
        super(AdversarialLoss, self).__init__()

    def forward(self, D_out_real, D_out_fake):
        self.epsilon = 1e-6

        D_out_real = torch.clamp(D_out_real, self.epsilon, 1.0 - self.epsilon)
        D_out_fake = torch.clamp(D_out_fake, self.epsilon, 1.0 - self.epsilon)

        # Compute adversarial loss
        adversarial_loss = torch.mean((torch.log(D_out_real)) + (torch.log(1 - D_out_fake)))
        return adversarial_loss

class AdjustedBCELoss(nn.Module):
    def __init__(self, lambda_=5, gamma=2):
        super(AdjustedBCELoss, self).__init__()
        self.lambda_ = lambda_
        self.gamma = gamma

    def forward(self, output, target, mask):

        # Loss function as defined in the paper
        bce_loss = F.binary_cross_entropy(output, target, reduction='none')
        adjusted_loss = ((self.lambda_*mask)*((output-target)**self.gamma))*bce_loss + \
                        (((1-mask)*(((output - target)**self.gamma)))*bce_loss)
        
        return torch.mean(adjusted_loss)
     
class TestBCELoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(TestBCELoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.criterion = nn.BCELoss()

    def __call__(self, output, is_real):
        labels = (self.real_label.to(device) if is_real else self.fake_label.to(device)).expand_as(output)
        loss = self.criterion(output, labels)
        return loss

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
        mask_files = os.listdir(self.mask_path)
        mask_name = random.choice(mask_files)
        
        img_path = os.path.join(self.image_path, file_name + '.jpg')
        mask_path = os.path.join(self.mask_path, mask_name)

        Im = cv2.imread(img_path, cv2.IMREAD_COLOR)
        M = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Random crop, scaling, or flip
        rand = random.randint(1, 2)
        Im = {
            1: lambda: Im,
            #2: lambda: utils.random_crop(Im),
            #3: lambda: utils.random_scaling(Im),
            2: lambda: rand_augmentation.flip(Im)
        }[rand]()

        Im2, Sm, gt_edge, M = image_processor_edge.process_image(Im, M)

        Im = cv2.resize(Im, (256, 256))

        if self.transform:
            Im2 = self.transform(Im2)
            Sm = self.transform(Sm)
            gt_edge = self.transform(gt_edge)
            M = self.transform(M)
            Im = self.transform(Im)

        return Im2, Sm, M, gt_edge, Im

if __name__ == "__main__":
    model = G2()
    model.to(device)
    D = Discriminator().to(device)

    g_loss = AdjustedBCELoss()
    d_loss_test = TestBCELoss()

    # Not sure why this is needed, but edge-connect uses it so it is here
    l1_loss = nn.L1Loss()

    g_optimizer = torch.optim.Adam(model.parameters(), betas=betas, lr=learning_rate)
    d_optimizer = torch.optim.Adam(D.parameters(), betas=betas, lr=d_learning_rate)

    scheduler_d = StepLR(d_optimizer, step_size=step_size_sc, gamma=gamma_sc)
    scheduler_g = StepLR(g_optimizer, step_size=step_size_sc, gamma=gamma_sc)
    
    transform = transforms.ToTensor()

    train_dataset = EdgeDataset(training_image_path, masks_path, img_amount, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = EdgeDataset(validation_image_path, masks_path, val_amount, transform=transform)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(num_epochs):
        # Training loop
        model.train()
        gen_running_loss = 0.0
        dis_running_loss = 0.0

        for Im, Sm, M, gt_edge, Im2 in train_dataloader:  
            gen_loss = 0
            dis_loss = 0

            d_optimizer.zero_grad()
            g_optimizer.zero_grad()

            Im = Im.to(device)
            Sm = Sm.to(device)
            M = M.to(device)
            gt_edge = gt_edge.to(device)
            Im2 = Im2.to(device)

            # Generate fake samples
            predicted = model(Im, Sm, M)

            # discriminator
            D_out_real, dis_real_feat = D(torch.cat((Im2, gt_edge), dim=1))
            D_out_fake, dis_fake_feat = D(torch.cat((Im2, predicted.detach()), dim=1))

            real_loss = d_loss_test(D_out_real, True)
            fake_loss = d_loss_test(D_out_fake, False)
            dis_loss = (real_loss + fake_loss) / 2

            # generator
            D_out_fake, gen_fake_feat = D(torch.cat((Im2, predicted), dim=1))
            g_gan_loss = d_loss_test(D_out_fake, True)
            gen_loss += g_gan_loss

            # edge-connect copy paste
            gen_fm_loss = 0
            for i in range(len(dis_real_feat)):
                gen_fm_loss += l1_loss(gen_fake_feat[i], dis_real_feat[i].detach())
            gen_fm_loss = gen_fm_loss * 10
            gen_loss += gen_fm_loss

            gen_running_loss += gen_loss
            dis_running_loss += dis_loss

            if dis_loss is not None:
                dis_loss.backward()
            d_optimizer.step()

            if gen_loss is not None:
                gen_loss.backward()
            g_optimizer.step()

            scheduler_d.step()
            scheduler_g.step()

        epoch_loss_adv = dis_running_loss / len(train_dataloader.dataset)
        epoch_loss_gen = gen_running_loss / len(train_dataloader.dataset)

        print(f"Epoch [{epoch+1}/{num_epochs}], Discriminator loss: {epoch_loss_adv:.4f}, Generator loss: {epoch_loss_gen:.4f}")

        torch.save(model.state_dict(), model_path)

        # Evaluation loop
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for Im_val, Sm_val, M_val, gt_edge_val, Im in val_dataloader:
                Im_val = Im_val.to(device)
                Sm_val = Sm_val.to(device)
                M_val = M_val.to(device)
                gt_edge_val = gt_edge_val.to(device)

                predicted_val = model(Im_val, Sm_val, M_val)
                val_loss = g_loss(predicted_val, gt_edge_val, M_val)
                val_running_loss += val_loss.item() * Im_val.size(0)

        val_epoch_loss = val_running_loss / len(val_dataloader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_epoch_loss:.4f}")

        # Break loop if the loss reaches the threshold
        if val_epoch_loss < min_loss_threshold:
            print("Validation loss is below the minimum threshold. Stopping training.")
            break



    