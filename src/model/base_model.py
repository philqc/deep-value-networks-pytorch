from abc import ABC, abstractmethod
import torch
from typing import Union, Tuple


class BaseModel(ABC):

    def __init__(self, metric_optimize: str, use_cuda: bool, label_dim: Union[int, Tuple[int]]):

        if metric_optimize.lower() == "f1":
            self.score_str = "F1"
            self.oracle_value = lambda x, y: self._f1_score(x, y)
        elif metric_optimize.lower() == "hamming":
            self.score_str = "Hamming"
            self.oracle_value = lambda x, y: self._scaled_hamming_loss(x, y)
        elif metric_optimize.lower() == "iou":
            self.score_str = "IOU"
            self.oracle_value = lambda x, y: self._iou_score(x, y)
        else:
            raise ValueError(f"Invalid metric_optimize provided = {metric_optimize}")

        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.label_dim = label_dim

        self.training = False

    @abstractmethod
    def train(self, loader):
        pass

    @abstractmethod
    def valid(self, loader):
        pass

    def _adjust_labels(self, pred_labels, gt_labels):
        if pred_labels.shape != gt_labels.shape:
            raise ValueError('Invalid labels shape: gt = ', gt_labels.shape, 'pred = ', pred_labels.shape)

        if not self.training:
            # No relaxation, 0-1 only
            pred_labels = torch.where(pred_labels >= 0.5,
                                      torch.ones(1).to(self.device),
                                      torch.zeros(1).to(self.device))
            pred_labels = pred_labels.float()
        return pred_labels, gt_labels

    def _f1_score(self, pred_labels, gt_labels):
        """
        Compute the ground truth value, i.e. v*(y, y*)
        of some predicted labels, where v*(y, y*)
        is the relaxed version of the F1 Score when training.
        and the discrete F1 when validating/testing
        """
        pred_labels, gt_labels = self._adjust_labels(pred_labels, gt_labels)

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

    def _iou_score(self, pred_labels, gt_labels):
        """
        Compute the ground truth value, i.e. v*(y, y*)
        of some predicted labels, where v*(y, y*)
        is the relaxed version of the IOU (intersection
        over union) when training, and the discrete IOU
        when validating/testing
        """
        pred_labels, gt_labels = self._adjust_labels(pred_labels, gt_labels)

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

    def _scaled_hamming_loss(self, pred_labels, gt_labels):
        """ Scaled Hamming Loss """
        pred_labels, gt_labels = self._adjust_labels(pred_labels, gt_labels)
        # Hamming Loss in 0-1
        loss = torch.sum(torch.abs(gt_labels - pred_labels), dim=1)
        if self.use_bce:
            # TODO: Change this constant
            loss /= 24.
        loss = loss.view(-1, 1)
        return loss
