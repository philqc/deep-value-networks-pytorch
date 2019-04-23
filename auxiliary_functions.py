import numpy as np
from torch.utils.data import Dataset
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import pickle
import os
import torch
from torchvision import utils
import numbers
from enum import Enum


class Sampling(Enum):
    GT = 1  # ground truth sampling
    ADV = 2  # adversarial sampling
    STRAT = 3  # stratified sampling


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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def show_img(img, black_and_white=True):
    np_img = img.numpy()
    # put channel at the end for plt.imshow
    if np_img.ndim == 3:
        np_img = np.transpose(np_img, (1, 2, 0))

    print('np_img.shape', np_img.shape)
    if black_and_white:
        plt.imshow(np_img, cmap='Greys_r')
        plt.show()
    else:
        plt.imshow(np_img)
        plt.show()


def save_img(img, path_to_save, black_and_white=True):
    np_img = img.numpy()
    np_img = np.transpose(np_img, (1, 2, 0))
    if black_and_white:
        plt.imsave(path_to_save + ".jpg", np_img, cmap='Greys_r')
    else:
        plt.imsave(path_to_save + ".jpg", np_img)


def save_grid_imgs(input_imgs, path_to_save, black_and_white=True):
    img = utils.make_grid(input_imgs, nrow=8)
    save_img(img, path_to_save, black_and_white)


def show_grid_imgs(input_imgs, black_and_white=True):
    img = utils.make_grid(input_imgs, nrow=8)
    show_img(img, black_and_white)


def print_a_sentence(x, y, txt_inputs, txt_labels):
    """ To visualize the data """
    print('-----------------')
    for i, x in enumerate(x):
        if x > 0.99:
            print(txt_inputs[i])
    
    print('-----------------')
    print('TAGS:')
    for i, y in enumerate(y):
        if y == 1:
            print(txt_labels[i])


def normalize_inputs(inputs, dir_path, load):
    if load:
        mean = np.load("%s/mean.npz.npy" % dir_path)
        std = np.load("%s/std.npz.npy" % dir_path)
    else:
        mean = np.mean(inputs, axis=0).reshape((1, -1))
        std = np.std(inputs, axis=0).reshape((1, -1)) + 10 ** -6

    train_inputs = inputs.astype(float)
    train_inputs -= mean
    train_inputs /= std

    if not load:
        np.save("%s/mean.npz" % dir_path, mean)
        np.save("%s/std.npz" % dir_path, std)
    return train_inputs


def compute_f1_score(labels, outputs):
    """ 
    Compute the example (total) (macro average) F1 measure
    """
    assert labels.shape == outputs.shape

    f1 = []
    for i in range(len(outputs)):
        f1.append(f1_score(labels[i], outputs[i]))

    return f1


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


def plot_results(results, iou=False):
    """
    Parameters:
    ----------
    results: dictionary with the train/valid loss
    and the f1 scores]
    iou: bool
      if true: print IOU, else print F1 Score
    """
    str_score = 'IOU' if iou else 'F1 Score'
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.set_title('Loss')
    ax1.set_ylabel('loss')
    ax1.set_xlabel('epochs')
    ax2.set_title('Validation ' + str_score)
    ax2.set_ylabel(str_score)
    ax2.set_xlabel('epochs')

    ax1.plot(results['loss_train'], label='loss_train')
    ax1.plot(results['loss_valid'], label='loss_valid')
    if iou:
        ax2.plot(results['IOU_valid'])
    else:
        ax2.plot(results['f1_valid'])

    ax1.legend()
    plt.show()


def plot_aggregate_results(results_path, iou, add_title=''):
    """
       Parameters:
       ----------
       iou: bool
         if true: print IOU, else print F1 Score
       """

    str_score = 'IOU' if iou else 'F1 Score'

    array_results = []
    for filename in os.listdir(results_path):
        if filename.endswith('.pkl'):
            with open(os.path.join(results_path, filename), 'rb') as fin:
                array_results.append(pickle.load(fin))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.set_title('Validation Loss ' + add_title)
    ax1.set_ylabel('loss')
    ax1.set_xlabel('epochs')
    ax2.set_title('Validation ' + str_score + ' ' + add_title)
    ax2.set_ylabel(str_score)
    ax2.set_xlabel('epochs')

    # max number of epochs
    max_ep = 375

    for res in array_results:

        if res['name'] == 'SPEN_bibtex':
            res['name'] = 'SPEN'

        label = res['name']

        # ax1.plot(res['loss_train'][:max_ep], label=label)
        if res['name'] != 'SPEN':
            ax1.plot(res['loss_valid'][:max_ep], label=label)

        if iou:
            ax2.plot(res['IOU_valid'][:max_ep], label=label)
        else:
            ax2.plot(res['f1_valid'][:max_ep], label=label)

    ax1.legend()
    ax2.legend()
    plt.show()


def plot_gradients(norm_grad, title):
    if len(norm_grad) == 0:
        print('No norm of gradients accumulated for {}'.format(title))
        return 0

    for i in range(len(norm_grad)):
        if i == 0 or (i + 1) % 30 == 0:
            plt.plot(norm_grad[i], label='{}'.format(i))
    plt.ylabel('Gradient norm')
    plt.xlabel('Steps')
    plt.title(title)
    plt.legend()
    plt.show()


def thirty_six_crop(img, size):
    """ Crop the given PIL Image 32x32 into 36 crops of 24x24
    Inspired from five_crop implementation in pytorch
    https://pytorch.org/docs/master/_modules/torchvision/transforms/functional.html
    """
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        assert len(size) == 2, "Please provide only two dimensions (h, w) for size."

    w, h = img.size
    crop_h, crop_w = size
    if crop_w > w or crop_h > h:
        raise ValueError("Requested crop size {} is bigger than input size {}".format(size, (h, w)))
    crops = []
    for i in [0, 2, 3, 4, 5, 8]:
        for j in [0, 2, 3, 4, 5, 8]:
            c = img.crop((i, j, crop_w + i, crop_h + j))
            if c.size != size:
                raise ValueError("Crop size is {} but should be {}".format(c.size, size))
            crops.append(c)

    return crops


def average_over_crops(pred, device):
    # batch_size x n_crops x height x width
    bs, n_crops, h, w = pred.shape
    final = torch.zeros(bs, 32, 32).to(device)
    size = torch.zeros(32, 32).to(device)
    for i, x in enumerate([0, 2, 3, 4, 5, 8]):
        for j, y in enumerate([0, 2, 3, 4, 5, 8]):
            k = i * 6 + j
            final[:, y:h + y, x: w + x] += pred[:, k].float()
            size[y:h + y, x: w + x] += 1

    final /= size
    final = final.view(bs, 1, 32, 32)
    return final


if __name__ == "__main__":

    # Path where the results in .pkl are stored
    directory_path = os.path.dirname(os.path.realpath(__file__))
    directory_path += '/saved_results/bibtex/'

    # Plot results from all the .pkl files
    plot_aggregate_results(directory_path, iou=False, add_title='Bibtex')
