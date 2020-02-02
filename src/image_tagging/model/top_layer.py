import torch.nn as nn


class TopLayer(nn.Module):
    """
    TopLayer from Graber & al. 2018:
    Deep Structured Prediction with Nonlinear Output Transformations
    """

    def __init__(self, label_dim):
        super().__init__()
        self.fc1 = nn.Linear(label_dim, 1152)
        self.fc2 = nn.Linear(1152, 1)

    def forward(self, y):
        y = nn.functional.hardtanh(self.fc1(y))
        y = self.fc2(y)
        return y
