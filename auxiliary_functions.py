import numpy as np
from torch.utils.data import Dataset
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import pickle
import os


class MyDataset(Dataset):

    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]

    def __len__(self):
        return len(self.inputs)


def print_a_sentence(x, y, txt_inputs, txt_labels):
    """ To visualize the data """
    print('-----------------')
    for i, x in enumerate(x):
        if x == 1:
            print(txt_inputs[i])
    
    print('-----------------')
    print('TAGS:')
    for i, y in enumerate(y):
        if y == 1:
            print(txt_labels[i])
    

def compute_f1_score(labels, outputs):
    """ 
    Compute the example averaged (macro average) F1 measure
    """
    assert labels.shape == outputs.shape

    f1 = []
    for i in range(len(outputs)):
        f1.append(f1_score(labels[i], outputs[i]))

    return np.mean(f1)


def plot_results(results):
    """
    Parameters:
    ----------
    results: dictionary with the train/valid loss
    and the f1 scores
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.set_title('Validation Loss')
    ax1.set_ylabel('loss')
    ax1.set_xlabel('epochs')
    ax2.set_title('Validation F1 Score')
    ax2.set_ylabel('F1 Score')
    ax2.set_xlabel('epochs')

    ax1.plot(results['loss_train'], label='loss_train')
    ax1.plot(results['loss_valid'], label='loss_valid')
    ax2.plot(results['f1_valid'])

    ax1.legend()
    plt.show()


def plot_aggregate_results(results_path, add_title=''):

    array_results = []
    for filename in os.listdir(results_path):
        if filename.endswith('.pkl'):
            with open(os.path.join(results_path, filename), 'rb') as fin:
                array_results.append(pickle.load(fin))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.set_title('Validation Loss ' + add_title)
    ax1.set_ylabel('loss')
    ax1.set_xlabel('epochs')
    ax2.set_title('Validation F1 Score ' + add_title)
    ax2.set_ylabel('accuracy')
    ax2.set_xlabel('epochs')

    # max number of epochs
    max_ep = 60

    for res in array_results:
        label = res['name']

        # ax1.plot(res['loss_train'][:max_ep], label=label)
        ax1.plot(res['loss_valid'][:max_ep], label=label)
        ax2.plot(res['f1_valid'][:max_ep], label=label)

    ax1.legend()
    ax2.legend()
    plt.show()


if __name__ == "__main__":

    # Path where the results in .pkl are stored
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path += '/saved_results/'

    # Plot results from all the .pkl files
    plot_aggregate_results(dir_path, 'Bibtex')
