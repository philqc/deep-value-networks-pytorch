import torch.nn as nn
import torch
import torchvision.models


class ConvNet(nn.Module):
    def __init__(self, label_dim):
        super().__init__()

        # Load pretrained AlexNet on ImageNet
        self.unary_model = torchvision.models.alexnet(pretrained=True)

        # Replace the last FC layer
        tmp = list(self.unary_model.classifier)
        tmp[-1] = nn.Linear(4096, label_dim)
        self.unary_model.classifier = nn.Sequential(*tmp)

    def forward(self, x):
        # send image through AlexNet
        x = self.unary_model(x)
        return torch.sigmoid(x)