import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import torchvision


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
    img = torchvision.utils.make_grid(input_imgs, nrow=8)
    save_img(img, path_to_save, black_and_white)


def show_grid_imgs(input_imgs, black_and_white=True):
    img = torchvision.utils.make_grid(input_imgs, nrow=8)
    show_img(img, black_and_white)


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
