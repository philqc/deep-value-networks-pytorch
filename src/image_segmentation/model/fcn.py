import torch.nn as nn
import torch.nn.functional as F


class FCN(nn.Module):

    # define each layer of neural network
    def __init__(self):
        super(FCN, self). __init__()
        # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv1 = nn.Conv2d(3, 64, 5, 1, padding=2)
        self.conv2 = nn.Conv2d(64, 128, 5, 2, padding=2)
        self.conv3 = nn.Conv2d(128, 128, 5, 2, padding=2)
        # Deconvolution
        # nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0,
        #                    output_padding=0, groups=1, bias=True, dilation=1)
        self.deconv1 = nn.ConvTranspose2d(128, 2, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.deconv1(x))
        x = self.deconv2(x)
        return x
