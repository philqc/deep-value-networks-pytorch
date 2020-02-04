import os
import random
from skimage import io
import numpy as np
from typing import Tuple
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF

from src.utils import project_root
from src.visualization_utils import show_img
from src.image_segmentation.utils import thirty_six_crop

PATH_DATA_WEIZMANN = os.path.join(project_root(), "data", "weizmann_horse")
PATH_SAVE_HORSE = os.path.join(project_root(), "saved_results", "weizmann_horse")


class WeizmannHorseDataset(Dataset):
    def __init__(self, img_dir: str, mask_dir: str, subset='train',
                 random_mirroring=True, thirty_six_cropping=False):
        """
        Args:
            img_dir(string): Path to the image file (training image)
            mask_dir(string): Path to the mask file (segmentation result)
            subset(string): 'train' or 'valid' or 'test'
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir

        self.img_names = os.listdir(img_dir)
        self.mask_names = os.listdir(mask_dir)

        self.img_names.sort()
        self.mask_names.sort()

        self.subset = subset
        if subset == 'test':
            self.img_names = self.img_names[200:]
            self.mask_names = self.mask_names[200:]
        elif subset == 'valid':
            self.img_names = self.img_names[180:200]
            self.mask_names = self.mask_names[180:200]
        else:
            self.img_names = self.img_names[:180]
            self.mask_names = self.mask_names[:180]

        self.transform = transforms.Compose([transforms.ToPILImage(),
                                             transforms.Resize(size=(32, 32))])
        self.random_mirroring = random_mirroring
        self.normalize = None
        self.thirty_six_cropping = thirty_six_cropping

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.img_names[idx])
        mask_name = os.path.join(self.mask_dir, self.mask_names[idx])

        image = io.imread(img_name)
        mask = io.imread(mask_name)

        image = self.transform(image)

        # create a channel for mask so as to transform
        mask = self.transform(np.expand_dims(mask, axis=2))

        if self.thirty_six_cropping:
            # Use 36 crops averaging for test set
            input_img = image

            image = thirty_six_crop(image, 24)
            if self.normalize is not None:
                transform_test = transforms.Compose(
                    [transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                     transforms.Lambda(lambda crops: torch.stack([self.normalize(crop) for crop in crops]))])
            else:
                transform_test = transforms.Compose(
                    [transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))])

            image = transform_test(image)
        else:
            # Random crop
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(24, 24))
            image = TF.crop(image, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)

            # Random Horizontal flipping
            if self.random_mirroring and random.random() > 0.50:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            input_img = image
            image = TF.to_tensor(image)
            if self.normalize is not None:
                image = self.normalize(image)

        input_img = TF.to_tensor(input_img)
        mask = TF.to_tensor(mask)

        # binarize mask again
        mask = mask >= 0.5
        if self.subset == "test":
            return input_img, image, mask
        else:
            return image, mask

    def compute_mean_and_stddev(self):
        n_images = len(self.img_names)
        masks, images = [], []

        # ToTensor transforms the images/masks in range [0, 1]
        transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize(size=(32, 32)),
                                        transforms.ToTensor()])

        for i in range(n_images):
            mask_name = os.path.join(self.mask_dir, self.mask_names[i])
            img_name = os.path.join(self.img_dir, self.img_names[i])
            mask = io.imread(mask_name)
            image = io.imread(img_name)

            image = transform(image)
            # create a channel for mask so as to transform
            mask = transform(np.expand_dims(mask, axis=2))

            # show_img(image, black_and_white=False)
            # show_img(mask.view(32, 32), black_and_white=True)
            masks.append(mask)
            images.append(image)

        # after torch.stack, we should have n_images x 1 x 32 x 32 for mask
        # and have n_images x 3 x 32 x 32 for images
        images, masks = torch.stack(images), torch.stack(masks)

        # compute mean and std_dev of images
        # put the images in n_images x 3 x (32x32) shape
        images = images.view(images.size(0), images.size(1), -1)
        mean_imgs = images.mean(2).sum(0) / n_images
        std_imgs = images.std(2).sum(0) / n_images

        # Find mean_mask for visualization purposes
        height, width = masks.shape[2], masks.shape[3]
        # flatten
        masks = masks.view(n_images, 1, -1)
        mean_mask = torch.mean(masks, dim=0)
        # go back to 32 x 32 view
        mean_mask = mean_mask.view(1, 1, height, width)

        print_mask = False
        if print_mask:
            img_to_show = mean_mask.squeeze(0)
            img_to_show = img_to_show.squeeze(0)
            show_img(img_to_show, black_and_white=True)

        return mean_imgs, std_imgs, mean_mask


def load_train_set_horse(
        path_data: str, use_cuda: bool, batch_size: int, batch_size_valid: int
) -> Tuple[DataLoader, DataLoader]:

    image_dir = os.path.join(path_data, 'images')
    mask_dir = os.path.join(path_data, 'masks')

    train_set = WeizmannHorseDataset(image_dir, mask_dir, subset='train',
                                     random_mirroring=False, thirty_six_cropping=False)
    valid_set = WeizmannHorseDataset(image_dir, mask_dir, subset='valid',
                                     random_mirroring=False, thirty_six_cropping=False)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        pin_memory=use_cuda
    )

    valid_loader = DataLoader(
        valid_set,
        batch_size=batch_size_valid,
        pin_memory=use_cuda
    )

    print(f'Using a {len(train_loader.dataset)}/{len(valid_loader.dataset)} train/validation split')

    return train_loader, valid_loader


def load_test_set_horse(
        path_data: str, use_cuda: bool, batch_size: int, thirtysix_crops: bool
) -> DataLoader:
    image_dir = os.path.join(path_data, 'images')
    mask_dir = os.path.join(path_data, 'masks')

    test_set = WeizmannHorseDataset(image_dir, mask_dir, subset='test',
                                    random_mirroring=False, thirty_six_cropping=thirtysix_crops)
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        pin_memory=use_cuda
    )
    return test_loader
