import arff
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import f1_score
import os
import random
from typing import Tuple

from src.utils import project_root, MyDataset

PATH_MODELS_ML_BIB = os.path.join(project_root(), "models", "multilabel_classification")
PATH_BIBTEX = os.path.join(project_root(), "data", "bibtex")


def get_bibtex(dir_path: str, use_train: bool):
    """
    Load the bibtex dataset.
    __author__ = "Michael Gygli, ETH Zurich"
    from https://github.com/gyglim/dvn/blob/master/mlc_datasets/__init__.py
    number of labels ("tags") = 159
    dimension of inputs = 1836
    Returns
    -------
    txt_labels (list)
        the 159 tags, e.g. 'TAG_system', 'TAG_social_nets'
    txt_inputs (list)
        the 1836 attribute words, e.g. 'dependent', 'always'
    labels (np.array)
        N x 159 array in one hot vector format
    inputs (np.array)
        N x 1839 array in one hot vector format
    """
    feature_idx = 1836
    if use_train:
        dataset = arff.load(open(os.path.join(dir_path, 'bibtex-train.arff'), 'r'))
    else:
        dataset = arff.load(open(os.path.join(dir_path, 'bibtex-test.arff'), 'r'))

    data = np.array(dataset['data'], np.int)

    labels = data[:, feature_idx:]
    inputs = data[:, 0:feature_idx]
    txt_labels = [t[0] for t in dataset['attributes'][feature_idx:]]
    txt_inputs = [t[0] for t in dataset['attributes'][:feature_idx]]
    return labels, inputs, txt_labels, txt_inputs


def load_training_set_bibtex(path_data: str, path_save: str, use_cuda: bool, batch_size=32,
                             norm_inputs=True, shuffle=True, train_valid_ratio=0.95
                             ) -> Tuple[DataLoader, DataLoader]:
    if train_valid_ratio < 0 or train_valid_ratio > 1:
        raise ValueError(f"Invalid train_valid_ratio = {train_valid_ratio} -> should be in [0, 1]")

    print('Loading the training set...')

    train_labels, train_inputs, txt_labels, txt_inputs = get_bibtex(path_data, use_train=True)
    if norm_inputs:
        train_inputs = normalize_inputs(train_inputs, path_save, load=False)
    train_data = MyDataset(train_inputs, train_labels)

    n_train = int(len(train_inputs) * train_valid_ratio)
    indices = list(range(len(train_inputs)))

    if shuffle:
        random.shuffle(indices)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(indices[:n_train]),
        pin_memory=use_cuda
    )
    valid_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(indices[n_train:]),
        pin_memory=use_cuda
    )

    print('Using a {} train {} validation split'.format(n_train, len(train_inputs) - n_train))
    return train_loader, valid_loader


def load_test_set_bibtex(path_data: str, path_save: str, use_cuda: bool) -> DataLoader:
    print('Loading Test set...')
    test_labels, test_inputs, txt_labels, txt_inputs = get_bibtex(path_data, use_train=False)
    test_inputs = normalize_inputs(test_inputs, path_save, load=True)
    test_data = MyDataset(test_inputs, test_labels)
    test_loader = DataLoader(
        test_data,
        batch_size=32,
        pin_memory=use_cuda
    )
    return test_loader


def print_a_sentence_bibtex(x, y, txt_inputs, txt_labels):
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


def normalize_inputs(inputs, path_save: str, load: bool):
    if load:
        mean = np.load(os.path.join(path_save, "mean.npz.npy"))
        std = np.load(os.path.join(path_save, "std.npz.npy"))
    else:
        mean = np.mean(inputs, axis=0).reshape((1, -1))
        std = np.std(inputs, axis=0).reshape((1, -1)) + 10 ** -6

    train_inputs = inputs.astype(float)
    train_inputs -= mean
    train_inputs /= std

    if not load:
        np.save(os.path.join(path_save, "mean.npz"), mean)
        np.save(os.path.join(path_save, "std.npz"), std)
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
