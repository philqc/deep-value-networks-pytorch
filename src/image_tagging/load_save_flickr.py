from torch.utils.data import DataLoader
import torch
import numpy as np
import os
from typing import Tuple

from src.image_tagging.model.conv_net import ConvNet
from src.utils import project_root
from src.image_tagging.flickr_dataset import (
    FlickrTaggingDataset, FlickrTaggingDatasetFeatures, inv_normalize, class_names
)
from src.visualization_utils import show_grid_imgs


PATH_FLICKR_DATA = os.path.join(project_root(), "data", "flickr")
PATH_SAVE_FLICKR = os.path.join(project_root(), "saved_results", "flickr")

TRAIN_LABEL_FILE = os.path.join(PATH_SAVE_FLICKR, 'train_labels_1k.pt')
VAL_LABEL_FILE = os.path.join(PATH_SAVE_FLICKR, 'val_labels.pt')
TRAIN_SAVE_IMG_FILE = os.path.join(PATH_SAVE_FLICKR, 'train_imgs_1k.pt')
VAL_SAVE_IMG_FILE = os.path.join(PATH_SAVE_FLICKR, 'val_imgs.pt')

TRAIN_FEATURE_FILE = os.path.join(PATH_SAVE_FLICKR, 'train_1k_features_20_epochs.pt')
VAL_FEATURE_FILE = os.path.join(PATH_SAVE_FLICKR, 'val_features_20_epochs.pt')

TRAIN_UNARY_FILE = os.path.join(PATH_SAVE_FLICKR, 'train_unary_20_epochs.pt')
VAL_UNARY_FILE = os.path.join(PATH_SAVE_FLICKR, 'val_unary_20_epochs.pt')


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
        load = True if TRAIN_FEATURE_FILE is not None else False
        print('Using Precomputed Unary Features!')
    elif use_unary:
        load = True if TRAIN_UNARY_FILE is not None else False
        print('Using Precomputed Unary Predictions!')
    else:
        load = True if TRAIN_SAVE_IMG_FILE is not None else False

    print('Loading training set....')
    if use_features or use_unary:
        feature_file = TRAIN_FEATURE_FILE if use_features else TRAIN_UNARY_FILE
        train_set = FlickrTaggingDatasetFeatures(type_dataset, images_folder=img_dir, feature_file=feature_file,
                                                 annotations_folder=label_dir, save_label_file=TRAIN_LABEL_FILE,
                                                 mode='train', load=load)
    else:
        train_set = FlickrTaggingDataset(type_dataset, img_dir, save_img_file=TRAIN_SAVE_IMG_FILE,
                                         annotations_folder=label_dir, save_label_file=TRAIN_LABEL_FILE,
                                         mode='train', load=load)

    print('Loading validation set....')
    if use_features or use_unary:
        feature_file = VAL_FEATURE_FILE if use_features else VAL_UNARY_FILE
        valid_set = FlickrTaggingDatasetFeatures(type_dataset, images_folder=img_dir, feature_file=feature_file,
                                                 annotations_folder=label_dir, save_label_file=VAL_LABEL_FILE,
                                                 mode='val', load=load)
    else:
        valid_set = FlickrTaggingDataset(type_dataset, img_dir, save_img_file=VAL_SAVE_IMG_FILE,
                                         annotations_folder=label_dir, save_label_file=VAL_LABEL_FILE,
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


def show_pred_labels(label, is_true_label):
    for i, l in enumerate(label):
        if is_true_label:
            if l > 0.99:
                print('{}, '.format(class_names[i]), end='')
        else:
            if l > 0.0001:
                print('{} = {:.0f}%, '.format(class_names[i], l * 100), end='')
                if i + 1 % 8 == 0 and i > 0:
                    print('')
    print('')


def visualize_predictions(inputs, pred_labels, targets, use_features: bool, use_unary: bool) -> None:
    idx = np.random.choice(np.arange(len(inputs)), 3, replace=False)
    if not use_features and not use_unary:
        inputs_unnormalized = inputs[idx].cpu()
        inputs_unnormalized = [inv_normalize(i) for i in inputs_unnormalized]
        show_grid_imgs(inputs_unnormalized, black_and_white=False)

    for i, j in enumerate(idx):
        print('({}) pred labels: '.format(i), end='')
        show_pred_labels(pred_labels[j], False)
        print('({}) true labels: '.format(i), end='')
        show_pred_labels(targets[j], True)
        print('------------------------------------')


def save_features(train_loader: DataLoader, valid_loader: DataLoader, path_model: str, label_dim=24):
    """Save features of the Unary Model code mainly taken from
    https://github.com/cgraber/NLStruct/blob/master/experiments/train_tagging_unary.py"""

    def save(model, dataloader, device, features_path):
        results = []
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            results.append(model(inputs).data.cpu())
        results = torch.cat(results)
        torch.save(results, features_path)

    def strip_classifier(model):
        tmp = list(model.classifier)
        tmp = tmp[:-1]
        model.classifier = torch.nn.Sequential(*tmp)

    only_features = True
    # If a GPU is available, use it
    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")
    conv = ConvNet(label_dim).to(device)
    conv.load_state_dict(torch.load(path_model))

    # Remove last layer of the model to store only the features
    if only_features:
        strip_classifier(conv.unary_model)

    conv.eval()

    if only_features:
        print('Saving Features for training data')
        save(conv, train_loader, device, TRAIN_FEATURE_FILE)
        print('Saving Features for validation data')
        save(conv, valid_loader, device, VAL_FEATURE_FILE)
    else:
        print('Saving Unary for training data')
        save(conv, train_loader, device, TRAIN_UNARY_FILE)
        print('Saving Unary for validation data')
        save(conv, valid_loader, device, VAL_UNARY_FILE)