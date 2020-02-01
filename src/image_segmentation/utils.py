import numbers
import torch


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