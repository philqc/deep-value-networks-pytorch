import arff
import numpy as np
from sklearn.metrics import f1_score


def get_bibtex(dir_path, split='train'):
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
    assert split in ['train', 'test']
    feature_idx = 1836
    if split == 'test':
        dataset = arff.load(open('%s/bibtex/bibtex-test.arff' % dir_path, 'r'))
    else:
        dataset = arff.load(open('%s/bibtex/bibtex-train.arff' % dir_path, 'r'))

    data = np.array(dataset['data'], np.int)

    labels = data[:, feature_idx:]
    inputs = data[:, 0:feature_idx]
    txt_labels = [t[0] for t in dataset['attributes'][feature_idx:]]
    txt_inputs = [t[0] for t in dataset['attributes'][:feature_idx]]

    if split == 'train':
        return labels, inputs, txt_labels, txt_inputs
    else:
        return labels, inputs, txt_labels, txt_inputs


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
