import matplotlib.pyplot as plt
import torch
from typing import List


def calculate_hamming_loss(true_labels: torch.Tensor, pred_labels: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.abs(true_labels - pred_labels))


def plot_hamming_loss(loss_train: List, loss_valid: List) -> None:
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.title('Hamming loss')
    plt.plot(loss_valid, label='hamming_loss_valid')
    plt.plot(loss_train, label='hamming_loss_train')
    plt.legend()
    plt.show()


