import numbers
import torch
from torch.utils.data import DataLoader
from typing import Tuple
import os
from src.utils import project_root
from src.visualization_utils import show_grid_imgs
from .weizmann_horse_dataset import WeizmannHorseDataset

PATH_DATA_WEIZMANN = os.path.join(project_root(), "data", "weizmann_horse")
PATH_SAVE_HORSE = os.path.join(project_root(), "saved_results", "weizmann_horse")


def load_train_set_horse(path_data: str, use_cuda: bool, batch_size: int,
                         batch_size_valid: int) -> Tuple[DataLoader, DataLoader]:

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

    print('Using a {} train {} validation split'
          ''.format(len(train_loader.dataset), len(valid_loader.dataset)))

    return train_loader, valid_loader


def load_test_set_horse(path_data: str, use_cuda: bool, batch_size: int,
                        thirtysix_crops: bool) -> DataLoader:

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


def get_iou_batch(y_pred, y_true):
    # extended domain oracle value function
    # define the oracle function for image segmentation(F1, IOU, or Dice Score) (tensor?)
    batch_size = y_pred.shape[0]

    scores = torch.zeros(batch_size, 1)
    for i in range(batch_size):
        scores[i] = get_iou(y_pred[i], y_true[i])

    return scores


def get_iou(y_pred, y_true):
    # y_pred and y_true are all "torch tensor"
    y_pred = torch.flatten(y_pred).reshape(1, -1)
    y_true = torch.flatten(y_true).reshape(1, -1)

    y_concat = torch.cat([y_pred, y_true], 0)

    intersect = torch.sum(torch.min(y_concat, 0)[0]).float()
    union = torch.sum(torch.max(y_concat, 0)[0]).float()
    return intersect / max(10 ** -8, union)


def show_preds_test_time(raw_inputs, final_pred, oracle) -> None:
    print('------ Test: IOU = {:.2f}% ------'.format(100 * oracle.mean()))
    img = raw_inputs.detach().cpu()
    show_grid_imgs(img)
    mask = final_pred.detach().cpu()
    show_grid_imgs(mask.float())
    print('Mask binary')
    bin_mask = mask >= 0.50
    show_grid_imgs(bin_mask.float())
    print('---------------------------------------')