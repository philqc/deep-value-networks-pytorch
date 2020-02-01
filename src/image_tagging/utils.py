import matplotlib.pyplot as plt
import torch


def calculate_hamming_loss(true_labels, pred_labels):
    loss = torch.sum(torch.abs(true_labels - pred_labels))
    return loss


def plot_hamming_loss(results):
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.title('Hamming loss')
    plt.plot(results['hamming_loss_valid'], label='hamming_loss_valid')
    plt.plot(results['hamming_loss_train'], label='hamming_loss_train')
    plt.legend()
    plt.show()
