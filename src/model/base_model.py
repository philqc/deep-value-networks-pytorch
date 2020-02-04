from abc import ABC, abstractmethod
import torch
from torch.utils.data import DataLoader


class BaseModel(ABC):

    def __init__(self, model):
        # If a GPU is available, use it
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.model = model.to(self.device)

    @abstractmethod
    def train(self, loader: DataLoader):
        pass

    @abstractmethod
    def valid(self, loader: DataLoader):
        pass

    @abstractmethod
    def test(self, loader: DataLoader):
        pass

    def _adjust_labels(self, pred_labels: torch.Tensor, gt_labels: torch.Tensor, training: bool):
        if pred_labels.shape != gt_labels.shape:
            raise ValueError('Invalid labels shape: gt = ', gt_labels.shape, 'pred = ', pred_labels.shape)

        if not training:
            # No relaxation, 0-1 only
            pred_labels = torch.where(pred_labels >= 0.5,
                                      torch.ones(1).to(self.device),
                                      torch.zeros(1).to(self.device))
            pred_labels = pred_labels.float()
        return pred_labels, gt_labels

    def _f1_score(self, pred_labels: torch.Tensor, gt_labels: torch.Tensor, training: bool):
        """
        Compute the ground truth value, i.e. v*(y, y*)
        of some predicted labels, where v*(y, y*)
        is the relaxed version of the F1 Score when training.
        and the discrete F1 when validating/testing
        """
        pred_labels, gt_labels = self._adjust_labels(pred_labels, gt_labels, training)

        intersect = torch.sum(torch.min(pred_labels, gt_labels), dim=1)
        union = torch.sum(torch.max(pred_labels, gt_labels), dim=1)

        # for numerical stability
        epsilon = torch.full(union.size(), 10 ** -8).to(self.device)

        # Add epsilon also in numerator since some images have 0 tags
        # and then we get 0/0 --> = nan instead of f1=1
        f1 = (2 * intersect + epsilon) / (intersect + torch.max(epsilon, union))
        # we want a (Batch_size x 1) tensor
        f1 = f1.view(-1, 1)
        return f1

    def _iou_score(self, pred_labels: torch.Tensor, gt_labels: torch.Tensor, training: bool):
        """
        Compute the ground truth value, i.e. v*(y, y*)
        of some predicted labels, where v*(y, y*)
        is the relaxed version of the IOU (intersection
        over union) when training, and the discrete IOU
        when validating/testing
        """
        pred_labels, gt_labels = self._adjust_labels(pred_labels, gt_labels, training)

        pred_labels = torch.flatten(pred_labels).reshape(pred_labels.size()[0], -1)
        gt_labels = torch.flatten(gt_labels).reshape(gt_labels.size()[0], -1)

        intersect = torch.sum(torch.min(pred_labels, gt_labels), dim=1)
        union = torch.sum(torch.max(pred_labels, gt_labels), dim=1)

        # for numerical stability
        epsilon = torch.full(union.size(), 10 ** -8).to(self.device)

        iou = intersect / torch.max(epsilon, union)
        # we want a (Batch_size x 1) tensor
        iou = iou.view(-1, 1)
        return iou