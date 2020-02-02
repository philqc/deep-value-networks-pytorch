from torch.utils.data import Dataset
import torch
from enum import Enum
from pathlib import Path
import os


class Sampling(Enum):
    GT = "Ground_Truth"
    ADV = "Adversarial"
    STRAT = "Stratified"


class Adam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.v = torch.zeros_like(params)
        self.m = torch.zeros_like(params)
        self.betas = betas
        self.lr = lr
        self.eps = eps
        self.t = 0

    def update(self, gradients):
        # update time step
        self.t += 1

        self.m = self.betas[0] * self.m + (1 - self.betas[0]) * gradients
        self.v = self.betas[1] * self.v + (1 - self.betas[1]) * (gradients ** 2)

        # Bias corrected first and second moment estimates
        mean = self.m / (1 - self.betas[0] ** self.t)
        variance = self.v / (1 - self.betas[1] ** self.t)

        update = self.lr * mean / (torch.sqrt(variance) + self.eps)
        return update


class SGD:

    def __init__(self, params, lr=0.5, momentum=0., weight_decay=0.0):
        """
        SGD with momentum for the inference part of
        training/testing. This speeds up training by using
        autograd.grad of the loss with respect to only the inputs
        but optimizer of pytorch is not compatible, so we make
        our own optimizer function
        """
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        self.lr = lr
        self.momentum = momentum
        self.v = torch.zeros_like(params)

    def update(self, gradients):
        self.v = self.momentum * self.v + self.lr * gradients
        return self.v


class MyDataset(Dataset):

    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]

    def __len__(self):
        return len(self.inputs)


def project_root() -> Path:
    """Returns project root folder."""
    return Path(__file__).parent.parent


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def create_path_that_doesnt_exist(path_save: str,  file_name: str, extension: str):
    if not os.path.isdir(path_save):
        os.makedirs(path_save)

    # Increment a counter so that previous results with the same args will not
    # be overwritten. Comment out the next four lines if you only want to keep
    # the most recent results.
    i = 0
    while os.path.exists(os.path.join(path_save, file_name + str(i) + extension)):
        i += 1

    return os.path.join(path_save, file_name + str(i) + extension)
