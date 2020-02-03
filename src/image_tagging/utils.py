import matplotlib.pyplot as plt
import torch
import os
from typing import Tuple
from torch.utils.data import DataLoader
from .load_flickr import FlickrTaggingDatasetFeatures, FlickrTaggingDataset


train_label_file = 'train_labels_1k.pt'
val_label_file = 'val_labels.pt'
train_save_img_file = 'train_imgs_1k.pt'
val_save_img_file = 'val_imgs.pt'

train_feature_file = 'train_1k_features_20_epochs.pt'
val_feature_file = 'val_features_20_epochs.pt'

train_unary_file = 'train_unary_20_epochs.pt'
val_unary_file = 'val_unary_20_epochs.pt'


def load_train_dataset_flickr(
        path_data: str,
        use_features: bool,
        use_unary: bool,
        use_cuda: bool,
        batch_size: int = 32,
        batch_size_eval: int = 512
) -> Tuple[DataLoader, DataLoader]:

    type_dataset = 'full'
    img_dir = os.path.join(path_data, 'mirflickr')
    label_dir = os.path.join(path_data, 'annotations')

    if use_unary and use_features:
        raise ValueError('Both using features and unary is impossible')
    if use_features:
        load = True if train_feature_file is not None else False
        print('Using Precomputed Unary Features!')
    elif use_unary:
        load = True if train_unary_file is not None else False
        print('Using Precomputed Unary Predictions!')
    else:
        load = True if train_save_img_file is not None else False

    print('Loading training set....')
    if use_features or use_unary:
        feature_file = train_feature_file if use_features else train_unary_file
        train_set = FlickrTaggingDatasetFeatures(type_dataset, images_folder=img_dir, feature_file=feature_file,
                                                 annotations_folder=label_dir, save_label_file=train_label_file,
                                                 mode='train', load=load)
    else:
        train_set = FlickrTaggingDataset(type_dataset, img_dir, save_img_file=train_save_img_file,
                                         annotations_folder=label_dir, save_label_file=train_label_file,
                                         mode='train', load=load)

    print('Loading validation set....')
    if use_features or use_unary:
        feature_file = val_feature_file if use_features else val_unary_file
        valid_set = FlickrTaggingDatasetFeatures(type_dataset, images_folder=img_dir, feature_file=feature_file,
                                                 annotations_folder=label_dir, save_label_file=val_label_file,
                                                 mode='val', load=load)
    else:
        valid_set = FlickrTaggingDataset(type_dataset, img_dir, save_img_file=val_save_img_file,
                                         annotations_folder=label_dir, save_label_file=val_label_file,
                                         mode='val', load=load)

    print(f'Using a {len(train_set)}/{len(valid_set)} train/validation split')

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        pin_memory=use_cuda
    )

    valid_loader = DataLoader(
        valid_set,
        batch_size=batch_size_eval,
        pin_memory=use_cuda
    )

    return train_loader, valid_loader


def calculate_hamming_loss(true_labels, pred_labels):
    return torch.sum(torch.abs(true_labels - pred_labels))


def plot_hamming_loss(results):
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.title('Hamming loss')
    plt.plot(results['hamming_loss_valid'], label='hamming_loss_valid')
    plt.plot(results['hamming_loss_train'], label='hamming_loss_train')
    plt.legend()
    plt.show()


