from torch.utils.data import Dataset
import torch
from pathlib import Path
import os


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


def create_path_that_doesnt_exist(path_save: str, file_name: str, extension: str):
    if not os.path.isdir(path_save):
        os.makedirs(path_save)
    # Increment a counter so that previous results with the same args will not
    # be overwritten. Comment out the next four lines if you only want to keep
    # the most recent results.
    i = 0
    while os.path.exists(os.path.join(path_save, file_name + str(i) + extension)):
        i += 1
    return os.path.join(path_save, file_name + str(i) + extension)
