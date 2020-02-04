import torch.nn as nn
import torch


class ConvNet(nn.Module):

    def __init__(self, non_linearity='relu'):
        super().__init__()
        # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv1 = nn.Conv2d(4, 64, 5, 1, padding=2)
        self.conv2 = nn.Conv2d(64, 128, 5, 2, padding=2)
        self.conv3 = nn.Conv2d(128, 128, 5, 2, padding=2)

        # Linear(in_features, out_features, bias=True)
        self.fc1 = nn.Linear(128 * 6 * 6, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, 1)

        # apply dropout on the first FC layer as paper mentioned
        self.dropout = nn.Dropout(p=0.25)

        non_linearity = non_linearity.lower()
        if non_linearity == 'softplus':
            self.non_linearity = nn.Softplus()
        elif non_linearity == 'relu':
            self.non_linearity = nn.ReLU()
        elif non_linearity == 'elu':
            self.non_linearity = nn.ELU()
        elif non_linearity == 'tanh':
            self.non_linearity = nn.Tanh()
        else:
            raise ValueError('Unknown activation Convnet:', non_linearity)

    def forward(self, x, y):
        # We first concatenate the img and the mask
        z = torch.cat((x, y), 1)

        z = self.non_linearity(self.conv1(z))
        z = self.non_linearity(self.conv2(z))
        z = self.non_linearity(self.conv3(z))

        # flatten before FC layers
        z = z.view(-1, 128 * 6 * 6)
        z = self.non_linearity(self.fc1(z))
        z = self.dropout(z)
        z = self.non_linearity(self.fc2(z))
        z = self.fc3(z)
        return z