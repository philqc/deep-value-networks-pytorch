import numpy as np
import matplotlib.pyplot as plt
import torchvision
from typing import Optional, List


def show_img(img, black_and_white=True, title: Optional[str] = None):
    np_img = img.numpy()
    # put channel at the end for plt.imshow
    if np_img.ndim == 3:
        np_img = np.transpose(np_img, (1, 2, 0))

    if title is not None:
        plt.title(title)

    if black_and_white:
        plt.imshow(np_img, cmap='Greys_r')
    else:
        plt.imshow(np_img)
    plt.show()


def save_img(img, path_to_save: str, black_and_white=True):
    np_img = img.numpy()
    np_img = np.transpose(np_img, (1, 2, 0))
    if black_and_white:
        plt.imsave(path_to_save + ".jpg", np_img, cmap='Greys_r')
    else:
        plt.imsave(path_to_save + ".jpg", np_img)


def save_grid_imgs(input_imgs, path_to_save: str, black_and_white=True):
    img = torchvision.utils.make_grid(input_imgs, nrow=8)
    save_img(img, path_to_save, black_and_white)


def show_grid_imgs(input_imgs, black_and_white=True, title: Optional[str] = None):
    img = torchvision.utils.make_grid(input_imgs, nrow=8)
    show_img(img, black_and_white, title)


def plot_results(str_score: str, train_loss: List, valid_loss: List,
                 valid_score: List, train_score: Optional[List] = None) -> None:
    """
    Parameters
    ----------
    str_score: Title for second plot (ex: F1 Score, Hamming Loss, IOU)
    train_loss: List of train loss accross epochs
    valid_loss
    valid_score
    train_score
    """
    x = list(range(1, len(train_loss) + 1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.set_title('Loss')
    ax1.set_ylabel('loss')
    ax1.set_xlabel('epochs')
    ax2.set_title(str_score)
    ax2.set_ylabel(str_score)
    ax2.set_xlabel('epochs')

    ax1.plot(x, train_loss, label='loss_train')
    ax1.plot(x, valid_loss, label='loss_valid')

    ax2.plot(x, valid_score, label="validation")
    if train_score is not None:
        ax2.plot(x, train_score, label="train")
        ax2.legend()

    ax1.legend()
    plt.show()
