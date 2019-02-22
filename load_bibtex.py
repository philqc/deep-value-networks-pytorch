import arff
import numpy as np


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
