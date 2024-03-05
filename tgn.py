import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torchvision.models import vgg19, VGG19_Weights


# Define the Encoder
class Encoder(nn.Module):
    def __init__(self, in_channels=4):
        super(Encoder, self).__init__()
        self.conv1 = spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=2,
                                             padding=0))
        self.in1 = nn.InstanceNorm2d(num_features=64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2))
        self.in2 = nn.InstanceNorm2d(num_features=128)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2))
        self.in3 = nn.InstanceNorm2d(num_features=256)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.in1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.in2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.in3(x)
        x = self.relu3(x)
        return x


# Define the Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = spectral_norm(nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=2,
                                             stride=1, dilation=2))
        self.in1 = nn.InstanceNorm2d(num_features=channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = spectral_norm(nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1))
        self.in2 = nn.InstanceNorm2d(num_features=channels)

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
        self.upconv1 = spectral_norm(nn.ConvTranspose2d(in_channels=in_channels, out_channels=128, kernel_size=3,
                                                        stride=2, padding=0))
        self.in1 = nn.InstanceNorm2d(num_features=128)
        self.relu1 = nn.ReLU(inplace=True)
        self.upconv2 = spectral_norm(nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2,
                                                        padding=0))
        self.in2 = nn.InstanceNorm2d(num_features=64)
        self.relu2 = nn.ReLU(inplace=True)
        self.upconv3 = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=3, stride=2, padding=0)
        self.in3 = nn.InstanceNorm2d(num_features=1)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.upconv1(x)
        x = self.in1(x)
        x = self.relu1(x)
        x = self.upconv2(x)
        x = self.in2(x)
        x = self.relu2(x)
        x = self.upconv3(x)
        x = self.in3(x)
        x = self.relu3(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = spectral_norm(nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=2, padding=0))
        self.conv2 = spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=0))
        self.conv3 = spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=0))
        self.conv4 = spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=0))
        self.conv5 = spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=2, padding=0))

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv3(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv4(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv5(x), negative_slope=0.2)
        return x


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder = Encoder(in_channels=4)
        self.residual_blocks = nn.ModuleList([ResidualBlock(256) for _ in range(8)])
        self.decoder = Decoder(in_channels=256)

    def forward(self, Im, Spred):
        # Concatenate the three images along the channel dimension
        combined_input = torch.cat((Im, Spred), dim=1)

        x = self.encoder(combined_input)
        for block in self.residual_blocks:
            x = block(x)
        x = self.decoder(x)
        x = torch.sigmoid(x)

        return x

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
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = self.gram_matrix(target_feature).clone().detach()

    @staticmethod
    def gram_matrix(layer):
        a, b, c, d = layer.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = layer.view(a * b, c * d)  # resize F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)

    def forward(self, layer):
        G = self.gram_matrix(layer)
        self.loss = F.mse_loss(G, self.target)
        return layer


# create a module to normalize input image so we can easily put it in a
# ``nn.Sequential``
class Normalization(nn.Module):
    def __init__(self):
        super(Normalization, self).__init__()
        # Normalization parameters for VGG networks
        cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
        cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])

        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(cnn_normalization_mean).view(-1, 1, 1)
        self.std = torch.tensor(cnn_normalization_std).view(-1, 1, 1)

    def forward(self, img):
        # normalize ``img``
        return (img - self.mean) / self.std


def get_style_model_and_loss(style_img):
    cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()

    # desired depth layers to compute style losses from the article
    style_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
    normalization = Normalization()

    # just in order to have an iterable access to or list of style losses
    style_losses = []

    # assuming that ``cnn`` is a ``nn.Sequential``, so we make a new ``nn.Sequential``
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    channel_sizes = set()
    i = 0  # increment every time we see a conv layer with new channel size
    j = 0  # increment every time we see a conv layer with current channel size
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            if layer.in_channels not in channel_sizes:
                i += 1
                j = 1
                channel_sizes.add(layer.in_channels)
            else:
                j += 1
            name = 'conv{}_{}'.format(i, j)
        elif isinstance(layer, nn.ReLU):
            name = 'relu{}_{}'.format(i, j)
            # The in-place version doesn't play very nicely with the ``ContentLoss``
            # and ``StyleLoss`` we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool{}_{}'.format(i, j)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn{}_{}'.format(i, j)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).clone().detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last style loss
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]
    # We also put the model in evaluation mode, so that specific layers
    # such as dropout or batch normalization layers behave correctly.
    model.eval()
    model.requires_grad_(False)

    print('model', model)
    print('style_losses', style_losses)

    return model, style_losses


def main():
    # Residual block structure
    # conv (dilation=2)
    # IN+ReLU
    # conv
    # IN

    pass


if __name__ == '__main__':
    main()
