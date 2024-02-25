import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


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

        # edge-connect left this last relu out so it is left out here also
        # out = self.relu(out)

        return out


# Generator same as G2 in PEGN
# Discriminator same as in PEGN

# total loss = lambda1 * L1_loss + lambda2 * adversarial_loss + lambda3 * style_loss

class L1Loss(nn.Module):
    def __init__(self, lambda_=5):
        super(L1Loss, self).__init__()
        self.lambda_ = lambda_

    def forward(self, output, target, mask):
        # Compute L1 loss

        # Loss function as defined in the paper
        L1_loss = F.l1_loss(output, target, reduction='none')
        adjusted_loss = self.lambda_ * np.absolute(mask * L1_loss) + np.absolute((1 - mask) * L1_loss)
        return torch.mean(adjusted_loss)


class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()

    @staticmethod
    def gram_matrix(feature):
        a, b, c, d = feature.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = feature.view(a * b, c * d)  # resize F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)

    def forward(self, output, target):
        # Compute style loss
        G_target = self.gram_matrix(target).detach()
        G_output = self.gram_matrix(output)
        style_loss = F.mse_loss(G_output, G_target)
        return style_loss


def main():
    # Residual block structure
    # conv (dilation=2)
    # IN+ReLU
    # conv
    # IN

    pass


if __name__ == '__main__':
    main()
