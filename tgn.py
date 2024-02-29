import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from torchvision.models import vgg19, VGG19_Weights


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
